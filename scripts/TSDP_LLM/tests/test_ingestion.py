import sys
import shutil
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import fitz  # PyMuPDF
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.chunker import TextChunker
from src.ingestion.vector_store import VectorStoreManager


def create_sample_football_pdf(output_path: Path) -> Path:
    """Creates a sample 2-page football tactics PDF for test verification."""
    doc = fitz.open()
    rect = fitz.Rect(50, 50, 550, 750)

    # Page 1: Tactical pressing
    page1 = doc.new_page()
    text_page1 = (
        "Tactical Periodization and High Pressing in Modern Football.\n\n"
        "Chapter 1: The Gegenpressing Concept.\n"
        "Gegenpressing is not an obsession with winning the ball back, but an instinct. "
        "When the opponent is in the transition phase after winning possession, they are at their most vulnerable. "
        "Their players are expanding into attacking shapes, leaving defensive gaps behind them. "
        "Immediate aggressive pressure applied within five seconds of ball loss forces turnovers high up the pitch, "
        "allowing the pressing team to strike immediately towards the opponent goal with fewer defenders in place."
    )
    page1.insert_textbox(rect, text_page1, fontsize=12)

    # Page 2: Build-up and positional play
    page2 = doc.new_page()
    text_page2 = (
        "Chapter 2: Positional Play (Juego de Posicion) in the First Phase of Build-up.\n"
        "The objective of Juego de Posicion is not merely ball retention, but creating numerical and positional superiority. "
        "The goalkeeper acts as an extra outfield player forming a 3+2 or 4+2 build-up structure. "
        "By drawing the opposing first line of pressure forward, passing lanes into the midfield pockets open up. "
        "The third-man concept is critical: player A passes to player B to find player C in space facing forward."
    )
    page2.insert_textbox(rect, text_page2, fontsize=12)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    doc.close()
    return output_path


def run_tests():
    print("🧪 Running Ingestion Pipeline Verification Tests...")
    test_dir = BASE_DIR / "tests" / "test_data"
    test_qdrant_dir = test_dir / "qdrant_test_db"
    test_pdf_path = test_dir / "sample_tactics_book.pdf"

    try:
        # Step 1: Create test PDF
        print("  1️⃣ Creating sample football tactics PDF...")
        create_sample_football_pdf(test_pdf_path)
        assert test_pdf_path.exists(), "PDF creation failed!"
        print(f"     ✅ Created test PDF: {test_pdf_path.name}")

        # Step 2: Test PDF Parser
        print("  2️⃣ Testing PDFParser...")
        parser = PDFParser()
        pages = parser.parse_pdf(test_pdf_path)
        assert len(pages) == 2, f"Expected 2 pages, got {len(pages)}"
        assert pages[0]["page_number"] == 1, "Page 1 number mismatch"
        assert "Gegenpressing" in pages[0]["text"], "Page 1 content missing keywords"
        assert pages[1]["page_number"] == 2, "Page 2 number mismatch"
        assert "Juego de Posicion" in pages[1]["text"], "Page 2 content missing keywords"
        print(f"     ✅ Successfully parsed {len(pages)} pages with correct metadata.")

        # Step 3: Test Chunker
        print("  3️⃣ Testing TextChunker...")
        chunker = TextChunker(target_chunk_size=50, chunk_overlap=15)
        chunks = chunker.chunk_book_pages(pages)
        assert len(chunks) >= 2, f"Expected at least 2 chunks, got {len(chunks)}"
        for chunk in chunks:
            assert "chunk_id" in chunk
            assert "start_page" in chunk
            assert "end_page" in chunk
            assert chunk["start_page"] in [1, 2]
        print(f"     ✅ Successfully generated {len(chunks)} chunks with exact page provenance.")

        # Step 4: Test VectorStore & FastEmbed
        print("  4️⃣ Testing VectorStoreManager (FastEmbed + Qdrant Local)...")
        vector_mgr = VectorStoreManager(
            db_path=test_qdrant_dir,
            collection_name="test_football_knowledge"
        )
        upserted_count = vector_mgr.upsert_chunks(chunks)
        assert upserted_count == len(chunks), f"Upserted count {upserted_count} != chunks count {len(chunks)}"
        print(f"     ✅ Successfully embedded and indexed {upserted_count} points in Qdrant.")

        # Step 5: Test Semantic Search
        print("  5️⃣ Testing Semantic Similarity Search...")
        query = "How does high pressing and counter-pressing work after losing possession?"
        results = vector_mgr.search(query, top_k=2, score_threshold=0.1)
        assert len(results) > 0, "Semantic search returned no results!"
        top_result = results[0]
        print(f"     🎯 Query: '{query}'")
        print(f"     📌 Top match (Score: {top_result['score']}): Book='{top_result['book_title']}', Page={top_result['start_page']}")
        print(f"     📝 Text snippet: {top_result['text'][:120]}...")
        assert top_result["start_page"] == 1, f"Expected page 1 for pressing topic, got {top_result['start_page']}"

        # Test Positional Play query
        query_2 = "What is the third-man concept in positional build-up play?"
        results_2 = vector_mgr.search(query_2, top_k=2, score_threshold=0.1)
        assert len(results_2) > 0, "Second search returned no results!"
        top_result_2 = results_2[0]
        print(f"     🎯 Query: '{query_2}'")
        print(f"     📌 Top match (Score: {top_result_2['score']}): Book='{top_result_2['book_title']}', Page={top_result_2['start_page']}")
        assert top_result_2["start_page"] == 2, f"Expected page 2 for build-up topic, got {top_result_2['start_page']}"

        print("\n✨ ALL TESTS PASSED SUCCESSFULLY! Phase 1 is fully functional.")

    finally:
        # Cleanup test files
        if test_dir.exists():
            try:
                # Qdrant client holds file lock on Windows, close client first if possible
                del vector_mgr
            except Exception:
                pass
            try:
                shutil.rmtree(test_dir, ignore_errors=True)
            except Exception:
                pass


if __name__ == "__main__":
    run_tests()
