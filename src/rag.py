"""
RAG pipeline for ticket triage.

Combines embedding-based retrieval with Gemini generation to:
1. Find similar historical tickets
2. Classify the new ticket
3. Generate a resolution suggestion based on retrieved context
"""

import json
import numpy as np
from typing import Dict, List

from .config import GEMINI_API_KEY, GENERATION_MODEL, TICKETS_PATH
from .embeddings import embed_text
from .vector_store import VectorStore
from .classifier import classify_ticket

RESOLUTION_PROMPT = """You are an oil & gas operations support system. A new support ticket has been submitted.

New ticket:
  Title: {title}
  Description: {description}

Here are similar historical tickets and how they were resolved:

{similar_tickets}

Based on the historical resolutions above, suggest a resolution approach for the new ticket.
Be specific, actionable, and concise (3-5 sentences). Reference the similar tickets when relevant.
"""


def _get_model():
    """Lazy-load the Gemini generative model."""
    import google.generativeai as genai

    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel(GENERATION_MODEL)


def _format_similar(tickets: List[Dict]) -> str:
    """Format similar tickets for the prompt."""
    lines = []
    for i, t in enumerate(tickets, 1):
        lines.append(f"--- Similar Ticket {i} (similarity: {t.get('score', 0):.2f}) ---")
        lines.append(f"Title: {t.get('title', 'N/A')}")
        lines.append(f"Category: {t.get('category', 'N/A')}")
        lines.append(f"Priority: {t.get('priority', 'N/A')}")
        lines.append(f"Resolution: {t.get('resolution', 'N/A')}")
        lines.append("")
    return "\n".join(lines)


def triage_ticket(
    title: str,
    description: str,
    store: VectorStore = None,
    top_k: int = 5,
) -> Dict:
    """
    Full RAG triage pipeline.

    1. Embed the query (title + description)
    2. Retrieve top_k similar historical tickets
    3. Classify category and priority via Gemini
    4. Generate resolution suggestion using retrieved context

    Parameters
    ----------
    title : str
        Ticket title.
    description : str
        Ticket description.
    store : VectorStore, optional
        Pre-loaded vector store. Loads from disk if None.
    top_k : int
        Number of similar tickets to retrieve.

    Returns
    -------
    dict
        Keys: classification, similar_tickets, resolution_suggestion, query.
    """
    # Step 1: Load store if needed
    if store is None:
        store = VectorStore.load()

    # Step 2: Embed the query and retrieve similar tickets
    query_text = f"{title}. {description}"
    query_embedding = np.array(embed_text(query_text))
    similar = store.search(query_embedding, top_k=top_k)

    # Step 3: Classify
    classification = classify_ticket(title, description)

    # Step 4: Generate resolution suggestion
    model = _get_model()
    prompt = RESOLUTION_PROMPT.format(
        title=title,
        description=description,
        similar_tickets=_format_similar(similar),
    )
    response = model.generate_content(prompt)

    return {
        "query": {"title": title, "description": description},
        "classification": classification,
        "similar_tickets": [
            {
                "id": t.get("id"),
                "title": t.get("title"),
                "category": t.get("category"),
                "priority": t.get("priority"),
                "resolution": t.get("resolution"),
                "score": t.get("score"),
            }
            for t in similar
        ],
        "resolution_suggestion": response.text,
    }
