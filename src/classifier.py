"""
Ticket classification using Google Gemini.

Classifies incoming tickets by category and priority using structured
prompts. Returns category, priority, confidence, and reasoning.
"""

import json
from typing import Dict

from .config import GEMINI_API_KEY, GENERATION_MODEL

VALID_CATEGORIES = [
    "equipment_failure",
    "production_decline",
    "safety_incident",
    "maintenance_request",
    "data_quality",
]
VALID_PRIORITIES = ["critical", "high", "medium", "low"]

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


def _get_model():
    """Lazy-load the Gemini generative model."""
    import google.generativeai as genai

    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel(GENERATION_MODEL)


def classify_ticket(title: str, description: str) -> Dict:
    """
    Classify a ticket's category and priority using Gemini.

    Returns
    -------
    dict
        Keys: category, priority, confidence, reasoning.
    """
    model = _get_model()
    prompt = CLASSIFY_PROMPT.format(
        title=title,
        description=description,
        categories=", ".join(VALID_CATEGORIES),
        priorities=", ".join(VALID_PRIORITIES),
    )
    response = model.generate_content(prompt)
    text = response.text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]  # remove first line
        text = text.rsplit("```", 1)[0]  # remove last fence
        text = text.strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        result = {
            "category": "unknown",
            "priority": "medium",
            "confidence": "low",
            "reasoning": f"Could not parse model response: {text[:200]}",
        }

    # Validate values
    if result.get("category") not in VALID_CATEGORIES:
        result["category"] = "unknown"
    if result.get("priority") not in VALID_PRIORITIES:
        result["priority"] = "medium"

    return result
