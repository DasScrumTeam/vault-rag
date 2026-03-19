#!/bin/bash
# Wrapper that auto-restarts vault-rag on crash (SIGSEGV, etc.)
# Used as MCP server command to survive ChromaDB Rust binding crashes.

MAX_RESTARTS=3
RESTART_WINDOW=60  # seconds - reset counter if stable for this long
RESTART_COUNT=0
LAST_RESTART=0

while true; do
    NOW=$(date +%s)

    # Reset counter if we've been stable long enough
    if (( NOW - LAST_RESTART > RESTART_WINDOW )); then
        RESTART_COUNT=0
    fi

    # Run vault-rag, passing all arguments through
    /Users/peterbeck/Obsidian/AME3/_Tools/vault-rag/.venv/bin/vault-rag "$@"
    EXIT_CODE=$?

    # Exit cleanly on normal termination (0, SIGTERM=143, SIGINT=130)
    if [[ $EXIT_CODE -eq 0 || $EXIT_CODE -eq 130 || $EXIT_CODE -eq 143 ]]; then
        exit $EXIT_CODE
    fi

    RESTART_COUNT=$((RESTART_COUNT + 1))
    LAST_RESTART=$(date +%s)

    if (( RESTART_COUNT >= MAX_RESTARTS )); then
        echo "vault-rag crashed $RESTART_COUNT times within ${RESTART_WINDOW}s, giving up." >&2
        exit $EXIT_CODE
    fi

    echo "vault-rag exited with code $EXIT_CODE, restarting ($RESTART_COUNT/$MAX_RESTARTS)..." >&2
    sleep 1
done
