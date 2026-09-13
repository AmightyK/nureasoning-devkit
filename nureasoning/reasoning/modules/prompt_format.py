"""Shared answer-format suffixes for eval, SFT, and nuVLA QA mix."""
from __future__ import annotations

import json
from typing import Any


def build_instruction_suffix(question_type: str, sample: dict[str, Any] | None = None) -> str:
    qt = (question_type or "").lower()
    ga = None
    if sample:
        ga = (sample.get("gold") or {}).get("answer")
    if qt == "choice" and isinstance(ga, list):
        return (
            "\n\nAnswer format: respond with ONLY the correct capital letters "
            "(e.g. A or A, C) for all that apply. No other text."
        )
    if qt == "choice":
        return (
            "\n\nAnswer format: respond with ONLY the single capital letter "
            '(A, B, C, or D) that matches the correct choice. No other text.'
        )
    if qt == "numerical" and isinstance(ga, list) and ga:
        first = ga[0]
        if isinstance(first, (list, tuple)) and len(first) >= 3:
            return (
                "\n\nAnswer format: respond with ONLY a JSON array "
                "[[t, x, y], ...] in meters, same length and time stamps as the question."
            )
        if len(ga) == 2:
            try:
                float(ga[0])
                float(ga[1])
                return (
                    "\n\nAnswer format: respond with ONLY a JSON array [x, y] in meters "
                    "or two numbers separated by a comma."
                )
            except (TypeError, ValueError):
                pass
    if qt == "numerical":
        return (
            "\n\nAnswer format: respond with ONLY the numeric value (one number). "
            "Use a plain decimal if needed. No units unless the question asks for them."
        )
    if qt == "categorical":
        return (
            "\n\nAnswer format: respond with ONLY the exact category label "
            "(same wording style as the reference). No other text."
        )
    return (
        "\n\nAnswer format: give a concise, direct answer matching the question. "
        "Do not use bullet lists unless the question asks."
    )


def format_question_prompt(question: dict[str, Any]) -> str:
    """User-turn text for VLM QA training and challenge answering."""
    text = str(question.get("question", "")).strip()
    choices = question.get("choices") or {}
    qtype = str(question.get("question_type", "open")).lower()
    lines = [
        "You are evaluating a driving scene. Answer the question based only "
        "on the provided camera observations.",
        "",
        f"Question: {text}",
    ]
    if qtype == "choice" and isinstance(choices, dict) and choices:
        lines.append("Choices:")
        for letter in sorted(choices.keys()):
            lines.append(f"  {letter}. {choices[letter]}")
    sample_like = {
        "question_type": qtype,
        "gold": {"answer": question.get("answer")},
    }
    lines.append(build_instruction_suffix(qtype, sample_like).lstrip("\n"))
    return "\n".join(lines)


def format_assistant_answer(question: dict[str, Any]) -> str:
    """Supervision string, formatted the way the challenge scorer parses it."""
    qtype = str(question.get("question_type") or "").lower()
    answer = question.get("answer")
    answer_text = question.get("answer_text")

    if qtype == "choice":
        if isinstance(answer, list):
            return ",".join(str(x).strip() for x in answer)
        if isinstance(answer, str):
            return answer.strip()
        if isinstance(answer_text, str):
            return answer_text.strip()
        return json.dumps(answer, ensure_ascii=False)

    if qtype == "numerical":
        if isinstance(answer, list):
            return json.dumps(answer, ensure_ascii=False)
        return "" if answer is None else str(answer)

    if isinstance(answer_text, str) and answer_text.strip():
        return answer_text.strip()
    if isinstance(answer, str):
        return answer.strip()
    if answer is not None:
        if isinstance(answer, (list, dict)):
            return json.dumps(answer, ensure_ascii=False)
        return str(answer)
    return ""
