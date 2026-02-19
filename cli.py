"""RAG Chat CLI — init, ingest, and dev server.

Usage:
    python cli.py init              Create knowledge/ dir and .env from template
    python cli.py ingest [DIR]      Index knowledge base into pgvector
    python cli.py dev               Start uvicorn with hot-reload
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("rag-cli")


def cmd_init(args):
    """Scaffold project: create knowledge/ dir, copy .env.example → .env."""
    root = Path(".")

    # Create knowledge dir
    kb = root / "knowledge"
    if not kb.exists():
        kb.mkdir()
        logger.info("[+] Created knowledge/ directory")
    else:
        logger.info("[=] knowledge/ already exists")

    # Copy .env.example → .env (if not exists)
    env_example = root / ".env.example"
    env_file = root / ".env"
    if not env_file.exists() and env_example.exists():
        shutil.copy(env_example, env_file)
        logger.info("[+] Created .env from .env.example — add your API key!")
    elif env_file.exists():
        logger.info("[=] .env already exists")
    else:
        logger.warning("[!] No .env.example found")

    # Check for example KB
    example = kb / "example.txt"
    if not example.exists() and not any(kb.glob("*.txt")):
        logger.info("[!] No knowledge base files found.  Add .txt/.md files to knowledge/")
    else:
        logger.info(f"[=] Knowledge base: {len(list(kb.glob('*')))} file(s)")

    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Edit .env with your LLM_API_KEY")
    logger.info("  2. Add knowledge base files to knowledge/")
    logger.info("  3. Start PostgreSQL: docker compose up postgres -d")
    logger.info("  4. Run: python cli.py dev")


def cmd_ingest(args):
    """Read knowledge base files, chunk, embed, store in pgvector."""
    from settings import settings

    kb_dir = Path(args.dir or settings.KNOWLEDGE_DIR)
    if not kb_dir.exists():
        logger.error(f"Directory not found: {kb_dir}")
        sys.exit(1)

    files = sorted(
        p for p in kb_dir.iterdir()
        if p.suffix in (".txt", ".md") and p.is_file()
    )
    if not files:
        logger.error(f"No .txt or .md files in {kb_dir}")
        sys.exit(1)

    logger.info(f"Indexing {len(files)} file(s) from {kb_dir}/")

    # Init DB
    import query_db
    if not query_db.init_db():
        logger.error("Database connection failed.  Is PostgreSQL running?")
        sys.exit(1)

    # Clear existing chunks
    query_db.clear_document_chunks()
    logger.info("Cleared existing document index")

    # Chunk and store
    chunk_size = settings.CHUNK_SIZE
    overlap = settings.CHUNK_OVERLAP
    stride = chunk_size - overlap
    total_chunks = 0

    for path in files:
        text = path.read_text(encoding="utf-8")
        chunks = [
            text[i : i + chunk_size]
            for i in range(0, len(text), stride)
            if text[i : i + chunk_size].strip()
        ]
        if chunks:
            query_db.store_document_chunks(chunks, source=path.name)
            total_chunks += len(chunks)
            logger.info(f"  {path.name}: {len(chunks)} chunks")

    logger.info(f"Done — {total_chunks} total chunks indexed in pgvector")


def cmd_dev(args):
    """Start uvicorn development server with hot-reload."""
    import subprocess

    from settings import settings

    host = args.host or settings.HOST
    port = args.port or settings.PORT

    logger.info(f"Starting dev server at http://{host}:{port}")
    subprocess.run(
        [
            sys.executable, "-m", "uvicorn",
            "main:app",
            "--host", host,
            "--port", str(port),
            "--reload",
        ],
        check=False,
    )


def main():
    parser = argparse.ArgumentParser(
        prog="rag",
        description="RAG Chat — policy-driven conversational AI framework",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # init
    sub.add_parser("init", help="Initialize project structure")

    # ingest
    p_ingest = sub.add_parser("ingest", help="Index knowledge base into pgvector")
    p_ingest.add_argument("dir", nargs="?", help="Knowledge base directory (default: knowledge/)")

    # dev
    p_dev = sub.add_parser("dev", help="Start development server")
    p_dev.add_argument("--host", help="Bind host")
    p_dev.add_argument("--port", type=int, help="Bind port")

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "dev":
        cmd_dev(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
