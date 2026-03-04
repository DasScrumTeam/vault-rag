"""Custom ObsidianReader with prefix filtering support."""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import yaml
from llama_index.core.schema import Document
from llama_index.readers.obsidian import ObsidianReader
from llama_index.readers.obsidian.base import is_hardlink
from shared.config import Config

logger = logging.getLogger(__name__)

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?\n)---\s*\n", re.DOTALL)


def extract_frontmatter_tags(content: str) -> str:
    """Extract tags from YAML frontmatter and return as comma-separated string.

    Handles both list and scalar forms. Returns "" on missing/invalid frontmatter.
    """
    fm = _parse_frontmatter(content)
    if not fm:
        return ""
    raw = fm.get("tags")
    if raw is None:
        return ""
    if isinstance(raw, list):
        return ",".join(str(t) for t in raw if t)
    return str(raw)


def extract_frontmatter_metadata(content: str) -> Dict[str, str]:
    """Extract all scalar frontmatter values as fm_-prefixed string dict.

    Lists are joined with commas. Nested dicts/complex values are skipped.
    Returns e.g. {"fm_quartopublish": "true", "fm_hugopublish": "true", "fm_tags": "book,definition"}.
    """
    fm = _parse_frontmatter(content)
    if not fm:
        return {}
    result: Dict[str, str] = {}
    for key, val in fm.items():
        if isinstance(val, list):
            # Join list items as comma-separated string
            result[f"fm_{key}"] = ",".join(str(v) for v in val if v)
        elif isinstance(val, bool):
            result[f"fm_{key}"] = str(val).lower()
        elif isinstance(val, (str, int, float)):
            result[f"fm_{key}"] = str(val)
        # Skip dicts and other complex types
    return result


def _parse_frontmatter(content: str) -> Optional[dict]:
    """Parse YAML frontmatter block. Returns dict or None."""
    m = _FRONTMATTER_RE.match(content)
    if not m:
        return None
    try:
        fm = yaml.safe_load(m.group(1))
        return fm if isinstance(fm, dict) else None
    except Exception:
        return None


def compute_folder_path(folder_name: str) -> str:
    """Return the vault-relative folder path (e.g. 'System/Rules/Arena')."""
    if not folder_name or folder_name == ".":
        return ""
    return folder_name


class ObsidianReaderWithFilter(ObsidianReader):
    """ObsidianReader with built-in prefix filtering support.

    This class extends the standard ObsidianReader to support filtering files
    based on filename prefixes before processing them. It preserves all
    Obsidian-specific features like task extraction, wikilinks, and backlinks.
    """

    def __init__(
        self,
        input_dir: str,
        config: Config,
        extract_tasks: bool = False,
        remove_tasks_from_text: bool = False,
    ):
        """Initialize the filtered ObsidianReader.

        Args:
            input_dir: Path to the Obsidian vault
            config: Configuration object containing prefix filters
            extract_tasks: Whether to extract tasks from documents
            remove_tasks_from_text: Whether to remove tasks from document text
        """
        super().__init__(input_dir, extract_tasks, remove_tasks_from_text)
        self.config = config

    def load_data(self, *args: Any, **load_kwargs: Any) -> List[Document]:
        """Load data with prefix filtering applied.

        This method overrides the parent's load_data to filter files by prefix
        before processing them, while maintaining all Obsidian-specific features.
        """
        docs: List[Document] = []
        # This map will hold: {target_note: [linking_note1, linking_note2, ...]}
        backlinks_map: dict[str, list[str]] = {}
        input_dir_abs = self.input_dir.resolve()

        # Debug logging
        logger.debug(f"Starting filtered document loading from: {self.input_dir}")
        logger.debug(f"Filter prefixes: {self.config.prefix_filter.allowed_prefixes}")

        processed_count = 0
        filtered_count = 0

        for dirpath, dirnames, filenames in os.walk(self.input_dir, followlinks=False):
            # Skip hidden directories.
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for filename in filenames:
                if filename.endswith(".md"):
                    processed_count += 1
                    # Apply prefix filtering here - this is the key change!
                    if not self.config.should_include_file(filename):
                        filtered_count += 1
                        logger.debug(f"Filtered out: {filename}")
                        continue  # Skip files that don't match the prefix filter
                    else:
                        logger.debug(f"Including: {filename}")

                    filepath = os.path.join(dirpath, filename)
                    file_path_obj = Path(filepath).resolve()
                    try:
                        if is_hardlink(filepath=file_path_obj):
                            print(
                                (
                                    f"Warning: Skipping file because it is a hardlink "
                                    f"(potential malicious exploit): {filepath}"
                                )
                            )
                            continue
                        if not str(file_path_obj).startswith(str(input_dir_abs)):
                            print(
                                (
                                    f"Warning: Skipping file outside input directory: "
                                    f"{filepath}"
                                )
                            )
                            continue
                        # Read raw file content to preserve exact character positions
                        with open(filepath, "r", encoding="utf-8") as f:
                            raw_content = f.read()

                        # Create a single document with the exact file content
                        md_docs = [Document(text=raw_content, metadata={})]
                        logger.debug(
                            f"File {filename} generated {len(md_docs)} documents"
                        )
                        for i, doc in enumerate(md_docs):
                            file_path_obj = Path(filepath)
                            note_name = file_path_obj.stem
                            doc.metadata["file_name"] = file_path_obj.name
                            doc.metadata["folder_path"] = str(file_path_obj.parent)
                            try:
                                folder_name = str(
                                    file_path_obj.parent.relative_to(input_dir_abs)
                                )
                            except ValueError:
                                # Fallback if relative_to fails (should not happen)
                                folder_name = str(file_path_obj.parent)
                            doc.metadata["folder_name"] = folder_name
                            doc.metadata["note_name"] = note_name
                            # Add file_path for compatibility with our system
                            doc.metadata["file_path"] = str(file_path_obj)
                            doc.metadata["tags"] = extract_frontmatter_tags(
                                raw_content
                            )
                            doc.metadata["folder"] = compute_folder_path(
                                folder_name
                            )
                            doc.metadata.update(
                                extract_frontmatter_metadata(raw_content)
                            )

                            wikilinks = self._extract_wikilinks(doc.text)
                            doc.metadata["wikilinks"] = wikilinks
                            # For each wikilink found in this document, record
                            #  a backlink from this note.
                            for link in wikilinks:
                                # Each link is expected to match a note name
                                # (without .md)
                                backlinks_map.setdefault(link, []).append(note_name)

                            # Optionally, extract tasks from the text.
                            if self.extract_tasks:
                                tasks, cleaned_text = self._extract_tasks(doc.text)
                                doc.metadata["tasks"] = tasks
                                if self.remove_tasks_from_text:
                                    md_docs[i] = Document(
                                        text=cleaned_text, metadata=doc.metadata
                                    )
                        docs.extend(md_docs)
                    except Exception as e:
                        print(f"Error processing file {filepath}: {e!s}")
                        continue

        # Now that we have processed all files, assign backlinks metadata.
        for doc in docs:
            doc_note_name: Optional[str] = cast(
                Optional[str],
                doc.metadata.get("note_name"),
            )
            # If no backlinks exist for this note, default to an empty list.
            if isinstance(doc_note_name, str):
                doc.metadata["backlinks"] = backlinks_map.get(doc_note_name, [])
            else:
                doc.metadata["backlinks"] = []

        # Debug logging
        logger.debug(
            (
                f"Processed {processed_count} .md files, "
                f"filtered out {filtered_count}, "
                f"loaded {len(docs)} documents"
            )
        )
        return docs
