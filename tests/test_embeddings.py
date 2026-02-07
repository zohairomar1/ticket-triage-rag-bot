"""Tests for the embeddings module."""

import json
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestEmbedText:
    def test_returns_list_of_floats(self):
        mock_result = {"embedding": [0.1, 0.2, 0.3] * 256}  # 768 dims

        with patch("src.embeddings._get_embed_fn") as mock_fn:
            mock_fn.return_value = MagicMock(return_value=mock_result)
            from src.embeddings import embed_text

            result = embed_text("test query")
            assert isinstance(result, list)
            assert len(result) == 768

    def test_calls_embed_fn_with_text(self):
        mock_result = {"embedding": [0.0] * 768}

        with patch("src.embeddings._get_embed_fn") as mock_fn:
            embed_fn = MagicMock(return_value=mock_result)
            mock_fn.return_value = embed_fn
            from src.embeddings import embed_text

            embed_text("pump failure")
            embed_fn.assert_called_once()
            call_kwargs = embed_fn.call_args
            assert "pump failure" in str(call_kwargs)


class TestEmbedTexts:
    def test_returns_numpy_array(self):
        mock_result = {"embedding": [[0.1] * 768, [0.2] * 768]}

        with patch("src.embeddings._get_embed_fn") as mock_fn:
            mock_fn.return_value = MagicMock(return_value=mock_result)
            from src.embeddings import embed_texts

            result = embed_texts(["text one", "text two"])
            assert isinstance(result, np.ndarray)
            assert result.shape == (2, 768)


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

            mock_result = {"embedding": [[0.1] * 768, [0.2] * 768]}

            with (
                patch("src.embeddings._get_embed_fn") as mock_fn,
                patch("src.embeddings.TICKETS_PATH", tickets_path),
                patch("src.embeddings.EMBEDDINGS_PATH", embeddings_path),
                patch("src.embeddings.TICKET_IDS_PATH", ids_path),
            ):
                mock_fn.return_value = MagicMock(return_value=mock_result)
                from src.embeddings import embed_tickets

                embed_tickets()

                assert embeddings_path.exists()
                assert ids_path.exists()

                loaded = np.load(embeddings_path)
                assert loaded.shape == (2, 768)

                with open(ids_path) as f:
                    ids = json.load(f)
                assert ids == ["TKT-001", "TKT-002"]
