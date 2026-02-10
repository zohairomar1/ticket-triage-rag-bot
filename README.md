# Ticket Triage RAG Bot

Automated support ticket triage for oil & gas operations. Classifies incoming tickets, finds similar historical issues, and suggests resolutions -- all powered by embeddings and retrieval-augmented generation.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)
![Tests](https://img.shields.io/badge/Tests-30%20Passing-brightgreen.svg)

---

## Why This Exists

Operations teams deal with a constant stream of support tickets -- pump trips, gas alarms, production drops, overdue inspections. Manually sorting through them to figure out what's urgent, what category it falls into, and how similar issues were fixed before is tedious and inconsistent. This project automates that entire workflow using RAG.

---

## Live Demo

[**Try it live on Streamlit Cloud**](https://ticket-triage-rag-bot.streamlit.app/)

https://github.com/user-attachments/assets/ticket-triage-demo.mov

https://user-images.githubusercontent.com/placeholder/ticket-triage-demo.mov

<video src="public/ticket-triage-rag-bot video.mov" controls width="100%"></video>

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | OpenAI GPT-4o Mini / Google Gemini 2.0 Flash |
| Embeddings | OpenAI text-embedding-3-small / Gemini gemini-embedding-001 |
| Vector Search | NumPy cosine similarity (no external vector DB) |
| Dashboard | Streamlit |
| Testing | Pytest with mocked API calls (30 tests) |

Dual-provider support -- switch between OpenAI and Gemini with a single env var.

---

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env   # Add your API key
python -m src.embeddings
streamlit run app/streamlit_app.py
```

---

## Future Improvements

- [ ] Hook into a real ticketing system (ServiceNow/Jira) so it can triage tickets as they come in rather than just demo data

---

## Contact

**Zohair Omar** -- [GitHub](https://github.com/zohairomar1)
