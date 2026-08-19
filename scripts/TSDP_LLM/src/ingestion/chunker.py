import re
import hashlib
from typing import List, Dict, Any


class TextChunker:
    """
    Chunks book text preserving page-level provenance (start_page, end_page)
    using word/sentence-based sliding window.
    """

    def __init__(self, target_chunk_size: int = 450, chunk_overlap: int = 80):
        self.target_chunk_size = target_chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        """Splits text into sentences while handling common abbreviations."""
        # Simple robust sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_book_pages(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunks pages belonging to a single book while tracking page numbers.
        Input format: [{'book_title': ..., 'page_number': 1, 'text': ...}, ...]
        Output format: [{
            'chunk_id': str,
            'book_title': str,
            'file_path': str,
            'start_page': int,
            'end_page': int,
            'chunk_index': int,
            'text': str,
            'word_count': int
        }, ...]
        """
        if not pages:
            return []

        book_title = pages[0]["book_title"]
        file_path = pages[0].get("file_path", "")

        # Build stream of sentences with their respective page numbers
        sentence_stream: List[Dict[str, Any]] = []
        for page in pages:
            p_num = page["page_number"]
            sents = self._split_into_sentences(page["text"])
            for s in sents:
                words = s.split()
                if not words:
                    continue
                sentence_stream.append({
                    "sentence": s,
                    "page_number": p_num,
                    "word_count": len(words)
                })

        if not sentence_stream:
            return []

        chunks: List[Dict[str, Any]] = []
        current_sentences: List[Dict[str, Any]] = []
        current_word_count = 0
        chunk_idx = 0
        i = 0

        while i < len(sentence_stream):
            item = sentence_stream[i]
            current_sentences.append(item)
            current_word_count += item["word_count"]

            # If chunk reached target size or at the end of text
            if current_word_count >= self.target_chunk_size or i == len(sentence_stream) - 1:
                chunk_text = " ".join(s["sentence"] for s in current_sentences)
                start_p = current_sentences[0]["page_number"]
                end_p = current_sentences[-1]["page_number"]

                # Generate deterministic chunk ID
                chunk_hash = hashlib.md5(f"{book_title}_{chunk_idx}_{start_p}_{chunk_text[:50]}".encode("utf-8")).hexdigest()[:12]
                chunk_id = f"{book_title}_c{chunk_idx}_{chunk_hash}"

                chunks.append({
                    "chunk_id": chunk_id,
                    "book_title": book_title,
                    "file_path": file_path,
                    "start_page": start_p,
                    "end_page": end_p,
                    "chunk_index": chunk_idx,
                    "text": chunk_text,
                    "word_count": len(chunk_text.split())
                })
                chunk_idx += 1

                if i == len(sentence_stream) - 1:
                    break

                # Prepare overlap for next window: keep tail sentences totaling ~chunk_overlap words
                overlap_sentences: List[Dict[str, Any]] = []
                overlap_words = 0
                for s in reversed(current_sentences):
                    if overlap_words + s["word_count"] <= self.chunk_overlap or not overlap_sentences:
                        overlap_sentences.insert(0, s)
                        overlap_words += s["word_count"]
                    else:
                        break

                current_sentences = list(overlap_sentences)
                current_word_count = overlap_words

            i += 1

        return chunks
