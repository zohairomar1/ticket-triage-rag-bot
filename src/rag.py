"""
RAG pipeline for ticket triage.

Combines embedding-based retrieval with LLM generation to:
1. Find similar historical tickets
2. Classify the new ticket
3. Generate a resolution suggestion based on retrieved context

Supports OpenAI and Gemini via LLM_PROVIDER config.
"""

import logging
import numpy as np
from typing import Dict, List

from .config import LLM_PROVIDER, API_KEY, GENERATION_MODEL

logger = logging.getLogger("rag")
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

_client = None


def _get_client():
    """Return a shared LLM client."""
    global _client
    if _client is None:
        if LLM_PROVIDER == "gemini":
            from google import genai
            _client = genai.Client(api_key=API_KEY)
        else:
            from openai import OpenAI
            _client = OpenAI(api_key=API_KEY)
    return _client


def _generate(client, prompt: str) -> str:
    """Generate text using the configured provider."""
    if LLM_PROVIDER == "gemini":
        response = client.models.generate_content(
            model=GENERATION_MODEL, contents=prompt
        )
        return response.text
    else:
        response = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content


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


def _fallback_resolution(similar: List[Dict]) -> str:
    """Template-based resolution when the LLM is unavailable."""
    if not similar:
        return "No similar historical tickets found. Manual review recommended."
    lines = [f"Based on {len(similar)} similar historical tickets:\n"]
    for i, t in enumerate(similar[:3], 1):
        lines.append(f"{i}. **{t.get('title', 'N/A')}** (similarity: {t.get('score', 0):.2f})")
        lines.append(f"   Resolution: {t.get('resolution', 'N/A')}\n")
    lines.append("Recommend following the resolution approach from the most similar ticket above.")
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
    3. Classify category and priority via LLM
    4. Generate resolution suggestion using retrieved context
    """
    if store is None:
        store = VectorStore.load()

    # Step 1: Embed and retrieve
    query_text = f"{title}. {description}"
    query_embedding = np.array(embed_text(query_text))
    similar = store.search(query_embedding, top_k=top_k)

    # Step 2: Classify
    classification = classify_ticket(title, description)

    # Step 3: Generate resolution
    try:
        client = _get_client()
        prompt = RESOLUTION_PROMPT.format(
            title=title,
            description=description,
            similar_tickets=_format_similar(similar),
        )
        resolution = _generate(client, prompt)
    except Exception as exc:
        logger.warning("LLM resolution failed, using template fallback: %s", str(exc)[:120])
        resolution = _fallback_resolution(similar)

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
        "resolution_suggestion": resolution,
    }
