"""
Embedding generation using Google Gemini text-embedding-004.

Run standalone: python -m src.embeddings
Generates: data/embeddings.npy, data/ticket_ids.json
"""

import json
import numpy as np
from typing import List

from .config import (
    GEMINI_API_KEY,
    TICKETS_PATH,
    EMBEDDINGS_PATH,
    TICKET_IDS_PATH,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
)


def _get_embed_fn():
    """Lazy-load the Gemini embedding function."""
    import google.generativeai as genai

    genai.configure(api_key=GEMINI_API_KEY)
    return genai.embed_content


def embed_text(text: str) -> List[float]:
    """Embed a single text string using Gemini."""
    embed_fn = _get_embed_fn()
    result = embed_fn(model=EMBEDDING_MODEL, content=text)
    return result["embedding"]


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed multiple texts and return as a numpy array."""
    embed_fn = _get_embed_fn()
    result = embed_fn(model=EMBEDDING_MODEL, content=texts)
    return np.array(result["embedding"])


def embed_tickets():
    """Embed all ticket descriptions and save to disk."""
    with open(TICKETS_PATH) as f:
        tickets = json.load(f)

    texts = [
        f"{t['title']}. {t['description']}" for t in tickets
    ]
    ids = [t["id"] for t in tickets]

    print(f"Embedding {len(texts)} tickets...")
    embeddings = embed_texts(texts)

    np.save(EMBEDDINGS_PATH, embeddings)
    with open(TICKET_IDS_PATH, "w") as f:
        json.dump(ids, f)

    print(f"Saved embeddings ({embeddings.shape}) -> {EMBEDDINGS_PATH}")
    print(f"Saved ticket IDs -> {TICKET_IDS_PATH}")


def load_embeddings():
    """Load cached embeddings and ticket IDs from disk."""
    embeddings = np.load(EMBEDDINGS_PATH)
    with open(TICKET_IDS_PATH) as f:
        ids = json.load(f)
    return embeddings, ids


if __name__ == "__main__":
    embed_tickets()
