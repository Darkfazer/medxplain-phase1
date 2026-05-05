"""
test_reformulation.py
=====================
Standalone script to test the medical query reformulation API.

Usage:
    export GEMINI_API_KEY=your_key_here
    python test_reformulation.py

If GEMINI_API_KEY is not set, the script exits with a warning.
"""
from __future__ import annotations

import os
import sys
import requests

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

REFORMULATION_PROMPT = (
    "You are a medical AI assistant specialized in radiology and histopathology. "
    "Rewrite the following non-expert question into a precise, medically accurate "
    "visual question answering (VQA) query suitable for a medical image analysis model. "
    "Add relevant medical terminology, focus on pathological findings, cellular abnormalities, "
    "and diagnostic features. Keep the reformulated question concise (under 50 words) "
    "and directly answerable from a medical image.\n\n"
    "Original question: {question}\n"
    "Reformulated question:"
)


def reformulate(question: str) -> str:
    if not GEMINI_API_KEY:
        print("WARNING: GEMINI_API_KEY not set. Skipping reformulation.")
        return question

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    )
    payload = {
        "contents": [{
            "parts": [{"text": REFORMULATION_PROMPT.format(question=question)}]
        }],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 128},
    }

    try:
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates", [])
        if candidates:
            text = (
                candidates[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
                .strip()
                .strip('"')
                .strip("'")
            )
            if text and len(text) > 5:
                return text
    except Exception as exc:
        print(f"ERROR: Reformulation failed: {exc}")
    return question


def main():
    test_questions = [
        "What is present?",
        "Is there something wrong with the lungs?",
        "What do you see in this image?",
        "Can you tell me about the heart?",
    ]

    for q in test_questions:
        print(f"\nOriginal:      {q}")
        result = reformulate(q)
        print(f"Reformulated:  {result}")


if __name__ == "__main__":
    main()
