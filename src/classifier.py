"""
Ticket classification with multi-provider support (OpenAI / Gemini).

Classifies incoming tickets by category and priority using structured
prompts. Returns category, priority, confidence, and reasoning.
"""

import json
from typing import Dict

from .config import LLM_PROVIDER, API_KEY, GENERATION_MODEL

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


def classify_ticket(title: str, description: str) -> Dict:
    """
    Classify a ticket's category and priority.

    Returns
    -------
    dict
        Keys: category, priority, confidence, reasoning.
    """
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
