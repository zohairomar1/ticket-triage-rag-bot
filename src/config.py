"""Configuration for the Ticket Triage RAG Bot."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

TICKETS_PATH = DATA_DIR / "tickets.json"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
TICKET_IDS_PATH = DATA_DIR / "ticket_ids.json"

EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-2.0-flash"

# Vector search defaults
DEFAULT_TOP_K = 5
EMBEDDING_DIMENSION = 3072
