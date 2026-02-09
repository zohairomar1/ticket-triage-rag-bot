"""Tests for the embeddings module."""

import json
import numpy as np
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock


def _mock_embed_result(vectors):
    """Build a mock embed_content result from a list of vectors."""
    embeddings = []
    for v in vectors:
        m = MagicMock()
        m.values = v
        embeddings.append(m)
    result = MagicMock()
    result.embeddings = embeddings
    return result


def _mock_client(vectors):
    """Build a mock Gemini client that returns the given vectors."""
    client = MagicMock()
    client.models.embed_content.return_value = _mock_embed_result(vectors)
    return client


class TestEmbedText:
    def test_returns_list_of_floats(self):
        client = _mock_client([[0.1, 0.2, 0.3] * 256])

        with (
            patch("src.embeddings.LLM_PROVIDER", "gemini"),
            patch("src.embeddings._get_client", return_value=client),
            patch("src.embeddings._cache_get", return_value=None),
            patch("src.embeddings._cache_put"),
        ):
            from src.embeddings import embed_text

            result = embed_text("test query")
            assert isinstance(result, list)
            assert len(result) == 768

    def test_cache_hit_skips_api_call(self):
        """When the cache has the embedding, no API call should be made."""
        cached_vec = np.array([0.5] * 768, dtype=np.float32)
        client = _mock_client([[0.0] * 768])

        with (
            patch("src.embeddings._get_client", return_value=client),
            patch("src.embeddings._cache_get", return_value=cached_vec),
        ):
            from src.embeddings import embed_text

            result = embed_text("test query")
            assert result == cached_vec.tolist()
            client.models.embed_content.assert_not_called()

    def test_cache_miss_calls_api(self):
        client = _mock_client([[0.1] * 768])

        with (
            patch("src.embeddings.LLM_PROVIDER", "gemini"),
            patch("src.embeddings._get_client", return_value=client),
            patch("src.embeddings._cache_get", return_value=None),
            patch("src.embeddings._cache_put"),
        ):
            from src.embeddings import embed_text

            embed_text("pump failure")
            client.models.embed_content.assert_called_once()


class TestEmbedTexts:
    def test_returns_numpy_array(self):
        client = _mock_client([[0.1] * 768, [0.2] * 768])

        with (
            patch("src.embeddings.LLM_PROVIDER", "gemini"),
            patch("src.embeddings._get_client", return_value=client),
            patch("src.embeddings._cache_get", return_value=None),
            patch("src.embeddings._cache_put"),
            patch("src.embeddings.EMBEDDING_DIMENSION", 768),
        ):
            from src.embeddings import embed_texts

            result = embed_texts(["text one", "text two"])
            assert isinstance(result, np.ndarray)
            assert result.shape == (2, 768)

    def test_partial_cache_only_embeds_uncached(self):
        """If 1 of 3 texts is cached, only 2 should be sent to the API."""
        cached_vec = np.array([0.9] * 768, dtype=np.float32)

        def cache_side_effect(text):
            if text == "cached text":
                return cached_vec
            return None

        client = _mock_client([[0.1] * 768, [0.2] * 768])

        with (
            patch("src.embeddings.LLM_PROVIDER", "gemini"),
            patch("src.embeddings._get_client", return_value=client),
            patch("src.embeddings._cache_get", side_effect=cache_side_effect),
            patch("src.embeddings._cache_put"),
            patch("src.embeddings.EMBEDDING_DIMENSION", 768),
        ):
            from src.embeddings import embed_texts

            result = embed_texts(["cached text", "new one", "new two"])
            assert result.shape == (3, 768)
            # First row should be the cached value
            np.testing.assert_array_equal(result[0], cached_vec)
            # API called once (one batch with the 2 uncached texts)
            client.models.embed_content.assert_called_once()

    def test_batching_splits_large_input(self):
        """Texts exceeding EMBEDDING_BATCH_SIZE should be split into multiple API calls."""
        batch_size = 2
        vecs = [[0.1] * 768] * 2  # each batch returns 2 vectors

        client = MagicMock()
        client.models.embed_content.return_value = _mock_embed_result(vecs)

        with (
            patch("src.embeddings.LLM_PROVIDER", "gemini"),
            patch("src.embeddings._get_client", return_value=client),
            patch("src.embeddings._cache_get", return_value=None),
            patch("src.embeddings._cache_put"),
            patch("src.embeddings.EMBEDDING_DIMENSION", 768),
            patch("src.embeddings.EMBEDDING_BATCH_SIZE", batch_size),
        ):
            from src.embeddings import embed_texts

            # 5 texts with batch_size=2 -> 3 API calls (2+2+1)
            # But we need to return correct number per batch
            call_count = [0]
            def side_effect(**kwargs):
                call_count[0] += 1
                contents = kwargs.get("contents", [])
                n = len(contents) if isinstance(contents, list) else 1
                return _mock_embed_result([[0.1] * 768] * n)

            client.models.embed_content.side_effect = side_effect
            result = embed_texts(["a", "b", "c", "d", "e"])
            assert result.shape == (5, 768)
            assert call_count[0] == 3  # ceil(5/2) = 3 batches


class TestEmbedTickets:
    def test_saves_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            tickets = [
                {"id": "TKT-001", "title": "Test", "description": "A test ticket."},
                {"id": "TKT-002", "title": "Test 2", "description": "Another ticket."},
            ]
            tickets_path = tmpdir / "tickets.json"
            embeddings_path = tmpdir / "embeddings.npy"
            ids_path = tmpdir / "ticket_ids.json"

            with open(tickets_path, "w") as f:
                json.dump(tickets, f)

            client = _mock_client([[0.1] * 768, [0.2] * 768])

            with (
                patch("src.embeddings.LLM_PROVIDER", "gemini"),
                patch("src.embeddings._get_client", return_value=client),
                patch("src.embeddings._cache_get", return_value=None),
                patch("src.embeddings._cache_put"),
                patch("src.embeddings.TICKETS_PATH", tickets_path),
                patch("src.embeddings.EMBEDDINGS_PATH", embeddings_path),
                patch("src.embeddings.TICKET_IDS_PATH", ids_path),
                patch("src.embeddings.EMBEDDING_DIMENSION", 768),
            ):
                from src.embeddings import embed_tickets

                embed_tickets()

                assert embeddings_path.exists()
                assert ids_path.exists()

                loaded = np.load(embeddings_path)
                assert loaded.shape == (2, 768)

                with open(ids_path) as f:
                    ids = json.load(f)
                assert ids == ["TKT-001", "TKT-002"]


class TestRateLimiter:
    def test_limiter_enforces_interval(self):
        from src.embeddings import _RateLimiter

        limiter = _RateLimiter(rpm=6000)  # 100 per second -> 0.01s interval
        limiter.wait()
        t0 = time.monotonic()
        limiter.wait()
        elapsed = time.monotonic() - t0
        # Should have waited ~0.01s (allow some tolerance)
        assert elapsed >= 0.005


class TestRetry:
    def test_retries_on_429(self):
        """_call_embed should retry on 429 errors."""
        from src.embeddings import _call_embed

        good_result = _mock_embed_result([[0.1] * 768])
        client = MagicMock()
        exc = Exception("429 RESOURCE_EXHAUSTED")
        client.models.embed_content.side_effect = [exc, good_result]

        with (
            patch("src.embeddings.LLM_PROVIDER", "gemini"),
            patch("src.embeddings._limiter") as mock_limiter,
        ):
            mock_limiter.wait = MagicMock()
            result = _call_embed(client, "test", max_retries=2)
            assert result == good_result
            assert client.models.embed_content.call_count == 2
