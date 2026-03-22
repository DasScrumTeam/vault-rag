"""Vector store management for document embeddings and semantic search."""

import logging
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union, cast

import chromadb
from chromadb.config import Settings
from components.embedding_system import EmbeddingModel, create_embedding_model
from components.vault_service.models import ChunkMetadata
from shared.config import EmbeddingModelConfig

logger = logging.getLogger(__name__)


def _preflight_check_collection(db_path: str, collection_name: str) -> bool:
    """Run a quick collection.count() in a subprocess.

    If the ChromaDB Rust bindings hit corrupt data they crash with SIGSEGV,
    which cannot be caught in-process.  By running the probe in a short-lived
    subprocess we can detect the crash and recover (delete + recreate) instead
    of taking down the whole MCP server.

    Returns True if the collection is healthy, False if it crashed or errored.
    """
    script = (
        "import sys, chromadb\n"
        "from chromadb.config import Settings\n"
        f"c = chromadb.PersistentClient(path={db_path!r},\n"
        "    settings=Settings(anonymized_telemetry=False, allow_reset=True))\n"
        "try:\n"
        f"    col = c.get_collection({collection_name!r})\n"
        "    col.count()\n"
        "except chromadb.errors.NotFoundError:\n"
        "    pass\n"  # Collection doesn't exist yet — that's fine
        # Any other exception (InternalError, etc.) propagates and
        # causes a non-zero exit code, which we treat as corruption.
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            timeout=30,
            capture_output=True,
        )
        if result.returncode != 0:
            logger.error(
                "ChromaDB preflight check failed (exit %d): %s",
                result.returncode,
                result.stderr.decode(errors="replace").strip()[-500:],
            )
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error("ChromaDB preflight check timed out.")
        return False
    except Exception as e:
        logger.error(f"ChromaDB preflight check failed: {e}")
        return False


class VectorStore:
    """Manages document embeddings and semantic search using ChromaDB."""

    def __init__(
        self,
        embedding_config: EmbeddingModelConfig,
        persist_directory: str = "./chroma_db",
        collection_name: str = "vault_docs",
    ):
        """Initialize the vector store.

        Args:
            embedding_config: Configuration for the embedding model
            persist_directory: Directory to persist the ChromaDB data
            collection_name: Name of the ChromaDB collection
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embedding_config = embedding_config

        # Lock to serialize all ChromaDB operations.
        # ChromaDB 1.x Rust bindings are not thread-safe for concurrent access
        # from multiple Python threads, causing SIGSEGV crashes.
        self._lock = threading.Lock()

        # Event signalling that initialization (and optionally the first
        # reindex) is complete.  Background threads that access the vector
        # store should call wait_until_ready() before their first operation.
        self._ready = threading.Event()

        # Ensure the persist directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # All ChromaDB operations during init are protected by the lock so
        # that any early background thread that slips through will block
        # rather than hit the Rust bindings concurrently.
        with self._lock:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )

        # Initialize the embedding model using the factory
        try:
            self.embedding_model = cast(
                EmbeddingModel, create_embedding_model(embedding_config)
            )
            logger.info(
                f"Initialized embedding model: "
                f"{embedding_config.provider}/{embedding_config.model_name}"
            )

            # Determine embedding dimension
            # We generate a dummy embedding to determine the dimension
            dummy_embedding = self.embedding_model.encode(["test_dimension_check"])[0]
            self.embedding_dimension = len(dummy_embedding)
            logger.debug(f"Detected embedding dimension: {self.embedding_dimension}")

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

        # Preflight: probe the collection in a subprocess so a SIGSEGV from
        # corrupt Rust index data doesn't kill the MCP server.
        if not _preflight_check_collection(
            str(self.persist_directory), self.collection_name
        ):
            logger.warning(
                "Corrupt collection detected. Deleting database and recreating."
            )
            with self._lock:
                self.client.reset()
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Obsidian vault document chunks"},
                )
                logger.info(
                    "Database reset and collection recreated after corruption."
                )
            return  # skip dimension check — collection is empty

        # Get or create the collection (under lock)
        with self._lock:
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"Loaded existing collection: {self.collection_name}")

                # Check for dimension mismatch if collection is not empty
                if self.collection.count() > 0:
                    result = self.collection.get(limit=1, include=["embeddings"])
                    if result["embeddings"] is not None and len(result["embeddings"]) > 0:
                        existing_dim = len(result["embeddings"][0])
                        if existing_dim != self.embedding_dimension:
                            logger.warning(
                                f"Dimension mismatch detected! Existing collection has dimension {existing_dim}, "
                                f"but current model produces {self.embedding_dimension}. "
                                f"Recreating collection '{self.collection_name}'..."
                            )
                            self.client.delete_collection(name=self.collection_name)
                            self.collection = self.client.create_collection(
                                name=self.collection_name,
                                metadata={"description": "Obsidian vault document chunks"},
                            )
                            logger.info(f"Recreated collection: {self.collection_name}")

            except chromadb.errors.NotFoundError:
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Obsidian vault document chunks"},
                )
                logger.info(f"Created new collection: {self.collection_name}")

    def mark_ready(self) -> None:
        """Signal that the vector store is ready for concurrent access."""
        self._ready.set()
        logger.info("VectorStore marked as ready.")

    def wait_until_ready(self, timeout: float | None = None) -> bool:
        """Block until the vector store is ready. Returns True if ready."""
        return self._ready.wait(timeout=timeout)

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Add document chunks to the vector store.

        Args:
            chunks: List of chunk dictionaries with text, file_path, chunk_id, and score
        """
        if not chunks:
            return

        # Extract texts and metadata
        texts = [chunk["text"] for chunk in chunks]
        chunk_ids = [chunk["chunk_id"] for chunk in chunks]

        # Create metadata for each chunk
        metadatas: List[Mapping[str, Union[str, int, float, bool, None]]] = []
        for chunk in chunks:
            meta: Dict[str, Union[str, int, float, bool, None]] = {
                "file_path": str(chunk["file_path"]),
                "score": float(chunk["score"]),
                "text_length": len(chunk["text"]),
                # Character offset metadata
                "start_char_idx": int(chunk.get("start_char_idx", 0)),
                "end_char_idx": int(chunk.get("end_char_idx", 0)),
                "original_text": str(chunk.get("original_text", "")),
                "document_id": str(chunk.get("document_id", "")),
                "tags": str(chunk.get("tags", "")),
                "folder": str(chunk.get("folder", "")),
            }
            # Include all fm_-prefixed frontmatter fields
            for key, val in chunk.items():
                if key.startswith("fm_"):
                    meta[key] = str(val)
            metadatas.append(meta)

        try:
            # Generate embeddings (CPU/network bound, safe outside lock)
            embeddings = self.embedding_model.encode(texts)

            # Add to ChromaDB in batches (ChromaDB max batch size is 5461)
            batch_size = 5000
            with self._lock:
                for i in range(0, len(chunks), batch_size):
                    end = min(i + batch_size, len(chunks))
                    self.collection.add(
                        embeddings=embeddings[i:end],  # type: ignore[arg-type]
                        documents=texts[i:end],
                        metadatas=metadatas[i:end],
                        ids=chunk_ids[i:end],
                    )
                    logger.debug(f"Added batch {i//batch_size + 1} ({end - i} chunks)")

            logger.debug(f"Added {len(chunks)} chunks to vector store")

        except Exception as e:
            logger.error(f"Error adding chunks to vector store: {e}")
            raise

    def search(
        self,
        query: str,
        limit: int = 5,
        quality_threshold: float = 0.0,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[ChunkMetadata]:
        """Search for relevant chunks using semantic similarity.

        Args:
            query: The search query
            limit: Maximum number of results to return
            quality_threshold: Minimum quality score for results
            where: Optional ChromaDB where clause for metadata filtering

        Returns:
            List of ChunkMetadata objects sorted by relevance
        """
        try:
            # Generate query embedding (CPU/network bound, safe outside lock)
            query_embedding = self.embedding_model.encode([query])[0]

            # Search the collection
            query_kwargs: Dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": limit * 2,  # Get more results to filter by quality
                "include": ["documents", "metadatas", "distances"],
            }
            if where is not None:
                query_kwargs["where"] = where
            with self._lock:
                results = self.collection.query(**query_kwargs)

            # Process results
            chunks = []
            if (
                results["documents"]
                and results["documents"][0]
                and results["metadatas"]
                and results["metadatas"][0]
                and results["distances"]
                and results["distances"][0]
            ):
                for i, (doc, metadata, distance) in enumerate(
                    zip(
                        results["documents"][0],
                        results["metadatas"][0],
                        results["distances"][0],
                        strict=False,
                    )
                ):
                    # 1. Filter by static quality score
                    static_quality_score = metadata.get("score", 0.0)
                    if static_quality_score is None:
                        static_quality_score = 0.0
                    if float(static_quality_score) < quality_threshold:
                        continue

                    # 2. Calculate relevance score for internal ranking
                    relevance_score = 1.0 - distance

                    chunk_id = (
                        results["ids"][0][i]
                        if results["ids"] and results["ids"][0]
                        else f"chunk_{i}"
                    )

                    # Create chunk with original quality score preserved
                    chunk = ChunkMetadata(
                        text=str(doc),
                        file_path=str(metadata.get("file_path", "")),
                        chunk_id=str(chunk_id),
                        score=float(
                            static_quality_score
                        ),  # Preserve original quality score
                        # Populate character offset fields from metadata
                        start_char_idx=int(metadata.get("start_char_idx") or 0),
                        end_char_idx=int(metadata.get("end_char_idx") or 0),
                        original_text=str(metadata.get("original_text") or ""),
                    )

                    # Store as tuple for sorting: (relevance_score, chunk)
                    chunks.append((relevance_score, chunk))

            # Sort by relevance score (first element of tuple) and extract chunks
            chunks.sort(key=lambda x: x[0], reverse=True)

            # Extract just the ChunkMetadata objects from the tuples
            return [chunk for _, chunk in chunks[:limit]]

        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []

    def remove_file_chunks(self, file_path: str) -> None:
        """Remove all chunks from a specific file.

        Args:
            file_path: Path of the file whose chunks should be removed
        """
        try:
            with self._lock:
                # Get all chunks from this file
                results = self.collection.get(
                    where={"file_path": file_path}, include=["metadatas"]
                )

                if results["ids"]:
                    # Delete the chunks
                    self.collection.delete(ids=results["ids"])
                    logger.info(f"Removed {len(results['ids'])} chunks from {file_path}")

        except Exception as e:
            logger.error(f"Error removing chunks for {file_path}: {e}")

    def get_all_file_paths(self) -> List[str]:
        """Get a list of all file paths that have chunks in the vector store.

        Returns:
            List of unique file paths
        """
        try:
            with self._lock:
                # Get all documents with metadata
                results = self.collection.get(include=["metadatas"])

            if results["metadatas"]:
                file_paths = {
                    str(metadata["file_path"])
                    for metadata in results["metadatas"]
                    if isinstance(metadata.get("file_path"), str)
                }
                return sorted(list(file_paths))

            return []

        except Exception as e:
            logger.error(f"Error getting file paths: {e}")
            return []

    def get_chunk_count(self) -> int:
        """Get the total number of chunks in the vector store.

        Returns:
            Total number of chunks
        """
        try:
            with self._lock:
                return self.collection.count()
        except Exception as e:
            logger.error(f"Error getting chunk count: {e}")
            return 0

    def clear_all(self) -> None:
        """Clear all data from the vector store."""
        try:
            with self._lock:
                # Delete the collection and recreate it
                self.client.delete_collection(name=self.collection_name)
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Obsidian vault document chunks"},
                )
            logger.info("Cleared all data from vector store")

        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            raise
