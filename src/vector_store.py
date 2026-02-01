"""
Lightweight vector store using numpy cosine similarity.

No external vector database required -- embeddings are stored as .npy files.
"""

import json
import numpy as np
from typing import Dict, List, Optional

from .config import TICKETS_PATH, EMBEDDINGS_PATH, TICKET_IDS_PATH, DEFAULT_TOP_K


class VectorStore:
    """In-memory vector store backed by numpy arrays."""

    def __init__(
        self,
        embeddings: Optional[np.ndarray] = None,
        ticket_ids: Optional[List[str]] = None,
        tickets: Optional[List[Dict]] = None,
    ):
        self.embeddings = embeddings
        self.ticket_ids = ticket_ids or []
        self._tickets_by_id: Dict[str, Dict] = {}
        if tickets:
            self._tickets_by_id = {t["id"]: t for t in tickets}

    @classmethod
    def load(cls) -> "VectorStore":
        """Load a pre-built store from disk."""
        embeddings = np.load(EMBEDDINGS_PATH)
        with open(TICKET_IDS_PATH) as f:
            ticket_ids = json.load(f)
        with open(TICKETS_PATH) as f:
            tickets = json.load(f)
        return cls(embeddings=embeddings, ticket_ids=ticket_ids, tickets=tickets)

    def search(
        self, query_embedding: np.ndarray, top_k: int = DEFAULT_TOP_K
    ) -> List[Dict]:
        """
        Find the top_k most similar tickets by cosine similarity.

        Parameters
        ----------
        query_embedding : np.ndarray
            1-D embedding vector for the query.
        top_k : int
            Number of results to return.

        Returns
        -------
        list of dict
            Each dict has ``id``, ``score``, and the full ticket fields.
        """
        if self.embeddings is None or len(self.embeddings) == 0:
            return []

        query = np.asarray(query_embedding, dtype=np.float32)

        # Cosine similarity: dot(a, b) / (||a|| * ||b||)
        norms = np.linalg.norm(self.embeddings, axis=1)
        query_norm = np.linalg.norm(query)

        # Avoid division by zero
        safe_norms = np.where(norms == 0, 1.0, norms)
        safe_query_norm = max(query_norm, 1e-10)

        similarities = self.embeddings @ query / (safe_norms * safe_query_norm)

        # Get top_k indices
        k = min(top_k, len(similarities))
        top_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_indices:
            ticket_id = self.ticket_ids[idx]
            ticket = self._tickets_by_id.get(ticket_id, {"id": ticket_id})
            results.append({
                **ticket,
                "score": float(similarities[idx]),
            })

        return results
