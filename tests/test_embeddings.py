"""Tests for the embeddings module."""

import json
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestEmbedText:
    def test_returns_list_of_floats(self):
        mock_embedding = MagicMock()
        mock_embedding.values = [0.1, 0.2, 0.3] * 256  # 768 dims
        mock_result = MagicMock()
        mock_result.embeddings = [mock_embedding]

        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = mock_result

        with patch("src.embeddings._get_client", return_value=mock_client):
            from src.embeddings import embed_text

            result = embed_text("test query")
            assert isinstance(result, list)
            assert len(result) == 768

    def test_calls_client_with_text(self):
        mock_embedding = MagicMock()
        mock_embedding.values = [0.0] * 768
        mock_result = MagicMock()
        mock_result.embeddings = [mock_embedding]

        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = mock_result

        with patch("src.embeddings._get_client", return_value=mock_client):
            from src.embeddings import embed_text

            embed_text("pump failure")
            mock_client.models.embed_content.assert_called_once()
            call_kwargs = mock_client.models.embed_content.call_args
            assert "pump failure" in str(call_kwargs)


class TestEmbedTexts:
    def test_returns_numpy_array(self):
        mock_e1 = MagicMock()
        mock_e1.values = [0.1] * 768
        mock_e2 = MagicMock()
        mock_e2.values = [0.2] * 768
        mock_result = MagicMock()
        mock_result.embeddings = [mock_e1, mock_e2]

        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = mock_result

        with patch("src.embeddings._get_client", return_value=mock_client):
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

            mock_e1 = MagicMock()
            mock_e1.values = [0.1] * 768
            mock_e2 = MagicMock()
            mock_e2.values = [0.2] * 768
            mock_result = MagicMock()
            mock_result.embeddings = [mock_e1, mock_e2]

            mock_client = MagicMock()
            mock_client.models.embed_content.return_value = mock_result

            with (
                patch("src.embeddings._get_client", return_value=mock_client),
                patch("src.embeddings.TICKETS_PATH", tickets_path),
                patch("src.embeddings.EMBEDDINGS_PATH", embeddings_path),
                patch("src.embeddings.TICKET_IDS_PATH", ids_path),
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
