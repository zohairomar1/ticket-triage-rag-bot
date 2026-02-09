"""Configuration for the Ticket Triage RAG Bot."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Provider selection: "openai" or "gemini"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

TICKETS_PATH = DATA_DIR / "tickets.json"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
TICKET_IDS_PATH = DATA_DIR / "ticket_ids.json"

# Model config per provider
if LLM_PROVIDER == "gemini":
    EMBEDDING_MODEL = "gemini-embedding-001"
    GENERATION_MODEL = "gemini-2.0-flash"
    EMBEDDING_DIMENSION = 3072
    API_KEY = GEMINI_API_KEY
else:
    EMBEDDING_MODEL = "text-embedding-3-small"
    GENERATION_MODEL = "gpt-4o-mini"
    EMBEDDING_DIMENSION = 1536
    API_KEY = OPENAI_API_KEY

# Vector search defaults
DEFAULT_TOP_K = 5

# Embedding cache
EMBEDDING_CACHE_DIR = DATA_DIR / "embedding_cache"

# Rate limiting
EMBEDDING_RPM_LIMIT = int(os.getenv("EMBEDDING_RPM_LIMIT", "90"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "20"))
