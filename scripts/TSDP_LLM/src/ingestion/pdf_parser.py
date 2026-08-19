import re
from pathlib import Path
from typing import List, Dict, Any
import fitz  # PyMuPDF


class PDFParser:
    """Extracts text from PDF books with page-level metadata."""

    @staticmethod
    def clean_text(text: str) -> str:
        """Cleans excessive whitespaces and formatting artifacts."""
        # Replace non-breaking spaces
        text = text.replace("\xa0", " ")
        # Replace multiple consecutive newlines with two
        text = re.sub(r"\n\s*\n", "\n\n", text)
        # Replace multiple spaces/tabs with single space
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def parse_pdf(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Parses a PDF file and returns a list of pages.
        Each item contains:
            - book_title: file name without extension
            - file_path: str
            - page_number: 1-indexed int
            - text: cleaned extracted text
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        book_title = file_path.stem
        pages: List[Dict[str, Any]] = []

        doc = fitz.open(file_path)
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            raw_text = page.get_text("text")
            cleaned = self.clean_text(raw_text)

            # Skip empty or negligible pages (e.g. blank cover/dividers)
            if len(cleaned) < 20:
                continue

            pages.append({
                "book_title": book_title,
                "file_path": str(file_path),
                "page_number": page_idx + 1,
                "text": cleaned
            })

        doc.close()
        return pages
