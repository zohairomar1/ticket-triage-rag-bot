"""
Ticket classification with multi-provider support (OpenAI / Gemini).

Classifies incoming tickets by category and priority using structured
prompts. Falls back to keyword matching when the LLM is unavailable.
"""

import json
import logging
from typing import Dict

from .config import LLM_PROVIDER, API_KEY, GENERATION_MODEL

logger = logging.getLogger("classifier")

VALID_CATEGORIES = [
    "equipment_failure",
    "production_decline",
    "safety_incident",
    "maintenance_request",
    "data_quality",
]
VALID_PRIORITIES = ["critical", "high", "medium", "low"]

_CATEGORY_KEYWORDS = {
    "equipment_failure": ["pump", "sensor", "valve", "motor", "trip", "leak", "failure",
                          "malfunction", "stuck", "broken", "esp", "pdg", "transmitter",
                          "seized", "corroded", "grinding", "washed out", "worn"],
    "production_decline": ["production", "decline", "dropped", "water cut", "gor",
                           "below forecast", "slugging", "wax", "underperforming",
                           "rate", "down", "oil rate", "water breakthrough"],
    "safety_incident": ["h2s", "alarm", "spill", "fire", "evacuat", "injury", "near miss",
                        "esd", "smoke", "lel", "dropped object", "slip", "fell", "psv lifted",
                        "safety", "loto"],
    "maintenance_request": ["schedule", "inspection", "annual", "overhaul", "calibration",
                            "certification", "replace", "maintenance", "pm", "due",
                            "order spares", "filter change"],
    "data_quality": ["data", "database", "missing", "duplicate", "inconsistent", "format",
                     "export", "dashboard", "scada", "timestamp", "etl", "validation"],
}

_PRIORITY_KEYWORDS = {
    "critical": ["shut in", "trip", "esd", "leak", "h2s", "zero", "offline", "fire",
                 "barrier", "evacuat", "shutdown"],
    "high": ["alarm", "dropping", "failed", "damaged", "urgent", "deadline", "overdue"],
    "low": ["annual", "routine", "preventive", "cosmetic", "administrative"],
}

CLASSIFY_PROMPT = """You are an oil & gas operations support system. Classify the following support ticket.

Ticket title: {title}
Ticket description: {description}

Classify into exactly ONE category from: {categories}
Assign exactly ONE priority from: {priorities}

Respond in valid JSON only, with these exact keys:
{{
  "category": "<one of the categories above>",
  "priority": "<one of the priorities above>",
  "confidence": "<high|medium|low>",
  "reasoning": "<one sentence explaining the classification>"
}}
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
            temperature=0.2,
        )
        return response.choices[0].message.content


def _classify_fallback(title: str, description: str) -> Dict:
    """Keyword-based classification when the LLM is unavailable."""
    text = f"{title} {description}".lower()

    # Score categories
    best_cat, best_score = "equipment_failure", 0
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > best_score:
            best_cat, best_score = cat, score

    # Score priorities
    priority = "medium"
    for pri in ["critical", "high", "low"]:
        if any(kw in text for kw in _PRIORITY_KEYWORDS[pri]):
            priority = pri
            break

    matched = [kw for kw in _CATEGORY_KEYWORDS.get(best_cat, []) if kw in text]
    reasoning = f"Keyword match: {', '.join(matched[:4])}" if matched else "Default classification"

    return {
        "category": best_cat,
        "priority": priority,
        "confidence": "medium",
        "reasoning": reasoning,
    }


def classify_ticket(title: str, description: str) -> Dict:
    """
    Classify a ticket's category and priority.

    Tries the LLM first; falls back to keyword matching on failure.

    Returns
    -------
    dict
        Keys: category, priority, confidence, reasoning.
    """
    try:
        client = _get_client()
        prompt = CLASSIFY_PROMPT.format(
            title=title,
            description=description,
            categories=", ".join(VALID_CATEGORIES),
            priorities=", ".join(VALID_PRIORITIES),
        )
        text = _generate(client, prompt).strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]
            text = text.strip()

        result = json.loads(text)

        # Validate values
        if result.get("category") not in VALID_CATEGORIES:
            result["category"] = "unknown"
        if result.get("priority") not in VALID_PRIORITIES:
            result["priority"] = "medium"

        return result
    except Exception as exc:
        logger.warning("LLM classification failed, using keyword fallback: %s", str(exc)[:120])
        return _classify_fallback(title, description)
