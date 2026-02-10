"""
Embedding generation using Google Gemini gemini-embedding-001.

Features:
- Singleton client (one genai.Client for the process lifetime)
- Disk-based embedding cache (keyed by model + text hash)
- Token-bucket rate limiter (default 90 RPM, configurable)
- Exponential backoff retry on 429 / transient errors
- Batch chunking for bulk embedding

Run standalone: python -m src.embeddings
Generates: data/embeddings.npy, data/ticket_ids.json
"""

import hashlib
import json
import logging
import time
import numpy as np
from typing import List, Optional

from .config import (
    GEMINI_API_KEY,
    TICKETS_PATH,
    EMBEDDINGS_PATH,
    TICKET_IDS_PATH,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    EMBEDDING_CACHE_DIR,
    EMBEDDING_RPM_LIMIT,
    EMBEDDING_BATCH_SIZE,
)

logger = logging.getLogger("embeddings")

# ---------------------------------------------------------------------------
# Singleton client
# ---------------------------------------------------------------------------

_client = None


def _get_client():
    """Return a shared Gemini client (created once per process)."""
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=GEMINI_API_KEY)
        logger.debug("Initialized Gemini client")
    return _client


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------

def _cache_key(text: str) -> str:
    """SHA-256 of model name + text. Includes model so cache invalidates on model change."""
    payload = f"{EMBEDDING_MODEL}:{EMBEDDING_DIMENSION}:{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_path(key: str):
    return EMBEDDING_CACHE_DIR / f"{key}.npy"


def _cache_get(text: str) -> Optional[np.ndarray]:
    """Return cached embedding or None."""
    key = _cache_key(text)
    path = _cache_path(key)
    if path.exists():
        logger.debug("cache hit  | key=%s… | len=%d", key[:12], len(text))
        return np.load(path)
    return None


def _cache_put(text: str, embedding: np.ndarray):
    """Write embedding to disk cache."""
    EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(text)
    np.save(_cache_path(key), embedding.astype(np.float32))


# ---------------------------------------------------------------------------
# Rate limiter (token bucket, 1 token = 1 API request)
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Simple token-bucket rate limiter for RPM."""

    def __init__(self, rpm: int):
        self.interval = 60.0 / max(rpm, 1)  # seconds between requests
        self._last = 0.0

    def wait(self):
        now = time.monotonic()
        elapsed = now - self._last
        if elapsed < self.interval:
            sleep_for = self.interval - elapsed
            logger.debug("rate-limit | sleeping %.2fs", sleep_for)
            time.sleep(sleep_for)
        self._last = time.monotonic()


_limiter = _RateLimiter(EMBEDDING_RPM_LIMIT)


# ---------------------------------------------------------------------------
# Retry with exponential backoff
# ---------------------------------------------------------------------------

def _call_embed(client, contents, max_retries: int = 4):
    """Call embed_content with retry on 429 / transient errors."""
    for attempt in range(max_retries + 1):
        _limiter.wait()
        try:
            result = client.models.embed_content(
                model=EMBEDDING_MODEL, contents=contents
            )
            return result
        except Exception as exc:
            status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            is_retryable = (
                "429" in str(exc)
                or "503" in str(exc)
                or "RESOURCE_EXHAUSTED" in str(exc)
                or status in (429, 503)
            )
            if is_retryable and attempt < max_retries:
                wait = min(2 ** attempt, 30)
                logger.warning(
                    "retry %d/%d | wait=%.1fs | error=%s",
                    attempt + 1, max_retries, wait, str(exc)[:120],
                )
                time.sleep(wait)
            else:
                raise


# ---------------------------------------------------------------------------
# Public API (signatures unchanged)
# ---------------------------------------------------------------------------

def embed_text(text: str, request_id: str = "") -> List[float]:
    """Embed a single text string using Gemini. Uses cache if available."""
    cached = _cache_get(text)
    if cached is not None:
        logger.info(
            "embed_text | cache=HIT | len=%d | req=%s", len(text), request_id
        )
        return cached.tolist()

    logger.info(
        "embed_text | cache=MISS | len=%d | req=%s", len(text), request_id
    )
    client = _get_client()
    result = _call_embed(client, text)
    embedding = result.embeddings[0].values

    _cache_put(text, np.array(embedding, dtype=np.float32))
    return embedding


def embed_texts(texts: List[str], request_id: str = "") -> np.ndarray:
    """Embed multiple texts and return as a numpy array.

    Checks the cache per-text first, then batches uncached texts
    into chunks of EMBEDDING_BATCH_SIZE for the API call.
    """
    n = len(texts)
    embeddings = np.zeros((n, EMBEDDING_DIMENSION), dtype=np.float32)
    uncached_indices = []

    # Check cache for each text
    for i, text in enumerate(texts):
        cached = _cache_get(text)
        if cached is not None:
            embeddings[i] = cached
        else:
            uncached_indices.append(i)

    cache_hits = n - len(uncached_indices)
    logger.info(
        "embed_texts | total=%d | cached=%d | to_embed=%d | req=%s",
        n, cache_hits, len(uncached_indices), request_id,
    )

    if not uncached_indices:
        return embeddings

    # Batch the uncached texts
    uncached_texts = [texts[i] for i in uncached_indices]
    client = _get_client()

    for batch_start in range(0, len(uncached_texts), EMBEDDING_BATCH_SIZE):
        batch = uncached_texts[batch_start:batch_start + EMBEDDING_BATCH_SIZE]
        batch_indices = uncached_indices[batch_start:batch_start + EMBEDDING_BATCH_SIZE]

        logger.info(
            "embed_texts | batch %d-%d of %d | req=%s",
            batch_start, batch_start + len(batch), len(uncached_texts), request_id,
        )

        result = _call_embed(client, batch)

        for j, (idx, emb) in enumerate(zip(batch_indices, result.embeddings)):
            vec = np.array(emb.values, dtype=np.float32)
            embeddings[idx] = vec
            _cache_put(uncached_texts[batch_start + j], vec)

    return embeddings


def embed_tickets():
    """Embed all ticket descriptions and save to disk.

    This is an OFFLINE command -- run via `python -m src.embeddings`.
    Never call this in a request path.
    """
    with open(TICKETS_PATH) as f:
        tickets = json.load(f)

    texts = [f"{t['title']}. {t['description']}" for t in tickets]
    ids = [t["id"] for t in tickets]

    print(f"Embedding {len(texts)} tickets...")
    embeddings = embed_texts(texts, request_id="bulk-index")

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
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
    embed_tickets()
