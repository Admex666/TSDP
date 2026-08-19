import argparse
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.config import settings
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.chunker import TextChunker
from src.ingestion.vector_store import VectorStoreManager


def ingest_file(
    file_path: Path,
    parser: PDFParser,
    chunker: TextChunker,
    vector_store: VectorStoreManager,
    reindex: bool = False
) -> int:
    """Ingests a single PDF into the vector store."""
    book_title = file_path.stem
    print(f"\n📖 Processing book: '{book_title}' ({file_path.name})...")

    if reindex:
        print(f"  🗑️ Removing previous index for '{book_title}'...")
        vector_store.delete_book(book_title)

    # 1. Parse PDF
    pages = parser.parse_pdf(file_path)
    if not pages:
        print(f"  ⚠️ No readable text found in '{file_path.name}'. Skipping.")
        return 0
    print(f"  ✅ Extracted text from {len(pages)} non-empty pages.")

    # 2. Chunk text
    chunks = chunker.chunk_book_pages(pages)
    if not chunks:
        print(f"  ⚠️ No chunks created for '{file_path.name}'. Skipping.")
        return 0
    print(f"  ✅ Generated {len(chunks)} text chunks (target size: {chunker.target_chunk_size} words).")

    # 3. Vector Embed & Store
    print(f"  ⏳ Generating embeddings and saving to Qdrant...")
    upserted = vector_store.upsert_chunks(chunks)
    print(f"  🎉 Successfully indexed {upserted} chunks for '{book_title}'.")
    return upserted


def main():
    parser = argparse.ArgumentParser(description="Ingest football literature PDFs into the vector knowledge base.")
    parser.add_argument("--file", type=str, default=None, help="Path to a specific PDF file to ingest.")
    parser.add_argument("--dir", type=str, default=None, help="Directory containing PDFs (default: data/books).")
    parser.add_argument("--reindex", action="store_true", help="Delete and recreate chunks for processed books.")
    args = parser.parse_args()

    pdf_parser = PDFParser()
    chunker = TextChunker(
        target_chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )
    vector_store = VectorStoreManager()

    if args.file:
        target_file = Path(args.file)
        if not target_file.exists():
            print(f"❌ Error: File not found: {target_file}")
            sys.exit(1)
        files = [target_file]
    else:
        target_dir = Path(args.dir) if args.dir else settings.BOOKS_DIR
        if not target_dir.exists():
            target_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(list(target_dir.glob("*.pdf")))

    if not files:
        print(f"ℹ️ No PDF files found in '{settings.BOOKS_DIR}'. Place football PDF books into 'data/books/' to ingest.")
        stats = vector_store.get_stats()
        print(f"Current Knowledge Base Stats: {stats}")
        return

    print(f"🚀 Starting Ingestion Pipeline: Found {len(files)} PDF file(s).")
    total_indexed = 0
    for f in files:
        try:
            total_indexed += ingest_file(f, pdf_parser, chunker, vector_store, reindex=args.reindex)
        except Exception as e:
            print(f"  ❌ Error processing '{f.name}': {e}")

    stats = vector_store.get_stats()
    print("\n==========================================")
    print("🎯 Ingestion Complete!")
    print(f"📊 Total chunks in vector base: {stats['total_chunks']}")
    print(f"🗃️ Qdrant Storage: {settings.QDRANT_PATH}")
    print("==========================================")


if __name__ == "__main__":
    main()
