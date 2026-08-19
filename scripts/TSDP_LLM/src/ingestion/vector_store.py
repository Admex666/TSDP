from pathlib import Path
from typing import List, Dict, Any, Optional
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from fastembed import TextEmbedding

from src.config import settings


class VectorStoreManager:
    """
    Manages embedded local Qdrant vector database and FastEmbed embedding generation.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        collection_name: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.db_path = str(db_path or settings.QDRANT_PATH)
        self.collection_name = collection_name or settings.COLLECTION_NAME
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME

        # Initialize embedded Qdrant client (local disk)
        self.client = QdrantClient(path=self.db_path)

        # Initialize FastEmbed (lightweight ONNX runtime)
        self.embedding_model = TextEmbedding(model_name=self.model_name)

        # Determine embedding dimension by testing a dummy token
        dummy_vector = list(self.embedding_model.embed(["test"]))[0]
        self.vector_dim = len(dummy_vector)

        self._ensure_collection()

    def _ensure_collection(self):
        """Creates the Qdrant collection if it does not already exist."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_dim,
                    distance=Distance.COSINE
                )
            )

    def upsert_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 64) -> int:
        """
        Generates embeddings for chunk texts and stores them in Qdrant with payload metadata.
        Returns total number of upserted points.
        """
        if not chunks:
            return 0

        total_upserted = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c["text"] for c in batch]

            # Generate embeddings via FastEmbed
            embeddings = list(self.embedding_model.embed(texts))

            points: List[PointStruct] = []
            for chunk, emb in zip(batch, embeddings):
                # Generate a stable UUID from chunk_id string
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["chunk_id"]))
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=emb.tolist() if hasattr(emb, "tolist") else list(emb),
                        payload={
                            "chunk_id": chunk["chunk_id"],
                            "book_title": chunk["book_title"],
                            "file_path": chunk.get("file_path", ""),
                            "start_page": chunk["start_page"],
                            "end_page": chunk["end_page"],
                            "chunk_index": chunk["chunk_index"],
                            "word_count": chunk.get("word_count", 0),
                            "text": chunk["text"],
                        }
                    )
                )

            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            total_upserted += len(points)

        return total_upserted

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.25,
        book_title: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs semantic cosine search for a query text.
        Returns top matching chunk payloads with similarity scores.
        """
        query_emb = list(self.embedding_model.embed([query]))[0]
        query_vector = query_emb.tolist() if hasattr(query_emb, "tolist") else list(query_emb)

        query_filter = None
        if book_title:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="book_title",
                        match=MatchValue(value=book_title)
                    )
                ]
            )

        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            score_threshold=score_threshold
        ).points

        results: List[Dict[str, Any]] = []
        for r in search_results:
            item = dict(r.payload) if r.payload else {}
            item["score"] = round(r.score, 4)
            results.append(item)

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Returns collection stats such as total points and indexed books."""
        try:
            info = self.client.get_collection(self.collection_name)
            points_count = info.points_count or 0
        except Exception:
            points_count = 0

        return {
            "collection_name": self.collection_name,
            "total_chunks": points_count,
            "vector_dimension": self.vector_dim,
            "embedding_model": self.model_name
        }

    def delete_book(self, book_title: str) -> None:
        """Deletes all chunks belonging to a specific book."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="book_title",
                        match=MatchValue(value=book_title)
                    )
                ]
            )
        )
