# Ticket Triage RAG Bot

A support ticket classifier and resolution suggester for oil & gas operations, powered by Gemini embeddings and retrieval-augmented generation (RAG).

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)
![Tests](https://img.shields.io/badge/Tests-25%20Passing-brightgreen.svg)

---

## Problem Statement

Operations teams in upstream oil & gas handle hundreds of support tickets -- equipment failures, production declines, safety incidents, maintenance requests, and data quality issues. Manual triage is slow and inconsistent.

This project demonstrates an automated triage system that:
- **Classifies** incoming tickets by category and priority using an LLM
- **Retrieves** similar historical tickets using embedding-based semantic search
- **Suggests** resolution approaches based on how similar issues were resolved

---

## How It Works

```
New Ticket
    |
    v
[Gemini Embedding] --> [Cosine Similarity Search] --> Similar Historical Tickets
    |                                                          |
    v                                                          v
[Gemini Classification]                              [Gemini Resolution Generator]
    |                                                          |
    v                                                          v
Category + Priority                               Suggested Resolution Approach
```

1. **Embed** the ticket title + description using Gemini `text-embedding-004`
2. **Retrieve** the top-k most similar tickets from the vector store (numpy cosine similarity)
3. **Classify** category and priority using Gemini 1.5 Flash with structured output
4. **Generate** a resolution suggestion using the retrieved tickets as context (RAG)

---

## What's Inside

| Component | Description |
|-----------|-------------|
| **Synthetic Dataset** | 100 realistic oil & gas support tickets across 5 categories |
| **Embedding Module** | Gemini text-embedding-004 with numpy caching |
| **Vector Store** | Lightweight cosine similarity search (no external DB) |
| **Classifier** | LLM-based category + priority classification with validation |
| **RAG Pipeline** | End-to-end retrieval-augmented generation for resolution suggestions |
| **Streamlit Dashboard** | Interactive demo with triage, browse, and search tabs |
| **Test Suite** | 25 unit tests with mocked Gemini API calls |

---

## Skills Demonstrated

- **RAG Architecture**: Embedding-based retrieval + LLM generation pipeline
- **Embeddings**: Gemini text-embedding-004 for semantic similarity
- **Vector Search**: Cosine similarity with numpy (no chromadb/pinecone dependency)
- **LLM Integration**: Structured prompting with JSON output parsing and validation
- **Synthetic Data Generation**: Domain-realistic ticket corpus for oil & gas operations
- **Testing**: Comprehensive mocked tests for all pipeline stages
- **Dashboard Development**: Streamlit with tabs, filters, and interactive search

---

## Ticket Categories

| Category | Count | Description |
|----------|-------|-------------|
| Equipment Failure | 22 | ESP trips, valve erosion, sensor faults, SCADA issues, crane faults, pump cavitation, turbine trips |
| Production Decline | 19 | Water breakthrough, GOR increase, wax plugging, liquid loading, sand production, forecast drift |
| Safety Incident | 16 | H2S alarms, dropped objects, ESD triggers, permit violations, falls, near-misses, fire events |
| Maintenance Request | 17 | BOP testing, compressor overhaul, flowline replacement, crane cert, valve greasing, SCADA patching |
| Data Quality | 16 | Missing data, unit mismatches, duplicate rows, MAPE issues, timestamp drift, allocation errors |

All tickets reference Volve field wellbores for continuity with the production analytics project.

---

## Quick Start

```bash
cd ticket-triage-ragbot

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set API key
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# Generate synthetic tickets (already included, but can regenerate)
python -m src.generate_tickets

# Generate embeddings (requires GEMINI_API_KEY)
python -m src.embeddings

# Launch dashboard
streamlit run app/streamlit_app.py
```

---

## Project Structure

```
ticket-triage-ragbot/
├── app/
│   └── streamlit_app.py          # Interactive demo dashboard
├── src/
│   ├── config.py                 # Paths, env vars, model settings
│   ├── generate_tickets.py       # Synthetic ticket generator
│   ├── embeddings.py             # Gemini embedding generation & caching
│   ├── vector_store.py           # NumPy cosine similarity search
│   ├── classifier.py             # LLM-based ticket classification
│   └── rag.py                    # Full RAG triage pipeline
├── data/
│   └── tickets.json              # 100 synthetic support tickets
├── tests/                        # 25 unit tests (all mocked)
│   ├── test_embeddings.py
│   ├── test_vector_store.py
│   ├── test_classifier.py
│   └── test_rag.py
├── requirements.txt
└── .env.example
```

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.9+ |
| LLM | Google Gemini 1.5 Flash |
| Embeddings | Gemini text-embedding-004 (768 dimensions) |
| Vector Search | NumPy cosine similarity |
| Dashboard | Streamlit |
| Testing | Pytest (mocked API calls) |

No LangChain. No ChromaDB. No Pinecone. Just numpy and the Gemini SDK.

---

## Design Decisions

1. **NumPy over vector DBs** -- 50 tickets fit in memory; cosine similarity in numpy is simple and transparent. No need for ChromaDB/Pinecone at this scale.
2. **Gemini 1.5 Flash** -- Fast, cheap, and capable enough for structured classification. The same model handles both classification and resolution generation.
3. **Structured JSON prompting** -- The classifier returns validated JSON with category, priority, confidence, and reasoning. Invalid responses are caught and defaulted.
4. **Embedding caching** -- Embeddings are saved as `.npy` files so they only need to be generated once. Queries are embedded on-the-fly.
5. **Mocked tests** -- All tests mock the Gemini API so they run without credentials and are deterministic.

---

## Interview Talking Points

- **RAG pattern**: "I built a retrieval-augmented generation pipeline from scratch: embed the query, retrieve similar documents via cosine similarity, then feed them to the LLM as context for generation. No LangChain -- I wanted to understand each step."

- **Embeddings**: "I use Gemini's text-embedding-004 model (768 dimensions) and cache the vectors as numpy arrays. The vector store is just cosine similarity over a matrix -- at 50 tickets, a vector database would be overengineering."

- **Classification**: "The classifier uses structured prompting to get JSON output from Gemini. I validate the response against allowed categories and priorities, with fallback defaults for malformed responses."

- **Testing strategy**: "All 25 tests mock the Gemini API calls. This means tests are deterministic, fast, and don't require API credentials. The mocks verify the pipeline structure and data flow, not the LLM output quality."

- **Domain data**: "I generated 50 realistic oil & gas support tickets covering 5 categories. The tickets reference the same Volve field wellbores used in my production analytics project, creating a coherent portfolio."

---

## Future Improvements

- [ ] Add evaluation metrics (precision, recall) against ground truth labels
- [ ] Implement re-ranking with cross-encoder after initial retrieval
- [ ] Add conversation memory for multi-turn ticket refinement
- [ ] Connect to a real ticketing system (ServiceNow, Jira) via API
- [ ] Scale to larger datasets with FAISS or ChromaDB

---

## Contact

**Zohair Omar** -- [GitHub](https://github.com/zohairomar1)
