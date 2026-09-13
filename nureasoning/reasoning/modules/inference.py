"""Async inference client and metric aggregation for reasoning evaluation.

Predictions are obtained from an OpenAI-compatible server (vLLM, SGLang, or a
hosted API). Requests are grouped by frame so that the images of a frame are
encoded once and, with vLLM prefix caching enabled, their KV cache is computed
once for all questions asked about that frame.
"""
from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from nureasoning.reasoning.modules.metrics import (
    compute_dataset_metrics,
    compute_metrics_for_sample,
    primary_metric_value,
)
from nureasoning.reasoning.modules.prompt_format import build_instruction_suffix

_b64_cache: dict[str, str] = {}


def clear_b64_cache() -> None:
    """Drop encoded images so a long run does not keep every clip in RAM."""
    _b64_cache.clear()


def file_to_data_url(path: str) -> str:
    cached = _b64_cache.get(path)
    if cached is not None:
        return cached
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode("ascii")
    url = f"data:{mime};base64,{b64}"
    _b64_cache[path] = url
    return url


def build_image_parts(
    image_paths: list[str],
    contexts: list[str] | None = None,
) -> list[dict[str, Any]]:
    use_ctx = isinstance(contexts, list) and len(contexts) == len(image_paths)
    parts: list[dict[str, Any]] = []
    for i, p in enumerate(image_paths):
        if not os.path.isfile(p):
            continue
        if use_ctx:
            text = str(contexts[i] or "").strip()
            if text:
                parts.append({"type": "text", "text": text})
        parts.append({"type": "image_url", "image_url": {"url": file_to_data_url(p)}})
    return parts


def build_user_content(
    sample: dict[str, Any],
    *,
    prebuilt_image_parts: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if prebuilt_image_parts is not None:
        parts = list(prebuilt_image_parts)
    else:
        parts = build_image_parts(
            sample.get("image_paths") or [],
            sample.get("image_contexts"),
        )
    qtext = sample.get("question") or ""
    choices = sample.get("choices")
    if isinstance(choices, dict) and choices:
        lines = [f"{k}) {v}" for k, v in sorted(choices.items())]
        qtext = qtext + "\n\nChoices:\n" + "\n".join(lines)
    qtext = qtext + build_instruction_suffix(str(sample.get("question_type")), sample)
    parts.append({"type": "text", "text": qtext})
    return parts


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _is_transient_api_failure(exc: BaseException) -> bool:
    """True for overload / transient server errors worth retrying."""
    code = getattr(exc, "status_code", None)
    if code is not None and code in (429, 500, 502, 503):
        return True
    name = type(exc).__name__
    if "Timeout" in name or "ConnectError" in name:
        return True
    text = str(exc).lower()
    if "timeout" in text or "connection reset" in text:
        return True
    if any(f"error code: {c}" in text for c in (429, 500, 502, 503)):
        return True
    return False


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Stratified means for every numeric metric, by question type / category."""
    metric_keys: set[str] = set()
    for r in rows:
        for k, v in (r.get("metrics") or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                metric_keys.add(k)

    by_type: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_cat: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_sub: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    pair_tc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    primary_by_type: dict[str, list[float]] = defaultdict(list)

    for r in rows:
        if r.get("error"):
            continue
        qt = str(r.get("question_type") or "unknown")
        cat = str(r.get("category") or "unknown")
        sub = str(r.get("subcategory") or "unknown")
        metrics = r.get("metrics") or {}
        pm = primary_metric_value(metrics, qt)
        if pm is not None:
            primary_by_type[qt].append(pm)

        for mk in metric_keys:
            v = metrics.get(mk)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                by_type[qt][mk].append(float(v))
                by_cat[cat][mk].append(float(v))
                by_sub[sub][mk].append(float(v))
                pair_tc[f"{qt}|{cat}"][mk].append(float(v))

    def summarize(nested: dict[str, dict[str, list[float]]]) -> dict[str, Any]:
        return {
            g: {k: {"mean": _mean(v), "count": len(v)} for k, v in mdict.items()}
            for g, mdict in nested.items()
        }

    return {
        "counts": {
            "total_rows": len(rows),
            "ok_rows": sum(1 for r in rows if not r.get("error")),
            "error_rows": sum(1 for r in rows if r.get("error")),
        },
        "primary_metric_mean_by_question_type": {
            k: _mean(v) for k, v in primary_by_type.items()
        },
        "dataset_metrics": compute_dataset_metrics(rows),
        "by_question_type": summarize(by_type),
        "by_category": summarize(by_cat),
        "by_subcategory": summarize(by_sub),
        "by_question_type_and_category": summarize(pair_tc),
    }


def _max_tokens_for(sample: dict[str, Any]) -> int:
    qt = str(sample.get("question_type") or "").lower()
    sub = str(sample.get("subcategory") or "").lower()
    if qt == "text":
        return 512
    if qt == "numerical" and "trajectory" in sub:
        return 2048
    return 128


async def run_one(
    client: AsyncOpenAI,
    sample: dict[str, Any],
    model: str,
    extra_body: dict[str, Any],
    *,
    max_retries: int = 3,
    retry_base_delay_s: float = 1.5,
    prebuilt_image_parts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    qtype = sample.get("question_type")
    base = {
        "sample_id": sample.get("sample_id"),
        "question_type": qtype,
        "category": sample.get("category"),
        "subcategory": sample.get("subcategory"),
    }
    t0 = time.perf_counter()
    for attempt in range(max_retries + 1):
        try:
            content = build_user_content(sample, prebuilt_image_parts=prebuilt_image_parts)
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_tokens=_max_tokens_for(sample),
                temperature=0.2,
                top_p=0.9,
                extra_body=extra_body,
            )
            msg = resp.choices[0].message
            pred = (msg.content or getattr(msg, "reasoning", None) or "").strip()
            gold = dict(sample["gold"])
            gold["choices"] = sample.get("choices")
            return {
                **base,
                "prediction_raw": pred,
                "latency_s": time.perf_counter() - t0,
                "metrics": compute_metrics_for_sample(str(qtype), pred, gold),
            }
        except Exception as e:  # noqa: BLE001
            if attempt < max_retries and _is_transient_api_failure(e):
                await asyncio.sleep(retry_base_delay_s * (2**attempt))
                continue
            return {**base, "error": str(e), "metrics": {}}
    return {**base, "error": "unreachable", "metrics": {}}


def group_by_frame(
    samples: list[dict[str, Any]],
) -> list[list[tuple[int, dict[str, Any]]]]:
    """Group samples that share the same image list, preserving manifest order."""
    groups: dict[tuple[tuple[str, ...], tuple[str, ...]], list[tuple[int, dict[str, Any]]]] = {}
    for i, s in enumerate(samples):
        key = (
            tuple(s.get("image_paths") or []),
            tuple(s.get("image_contexts") or []),
        )
        groups.setdefault(key, []).append((i, s))
    return list(groups.values())


async def run_inference(
    samples: list[dict[str, Any]],
    base_urls: list[str],
    model: str,
    concurrent_per_url: int,
    api_key: str,
    thinking: bool,
    *,
    max_retries: int = 3,
    pipeline_depth: int = 2,
    predictions_path: Path | None = None,
    clients: list[Any] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate every sample and return predictions in manifest order.

    Predictions are streamed to *predictions_path* as soon as a contiguous
    prefix is complete, so a crashed run leaves usable partial output.
    """
    if not base_urls:
        raise ValueError("No API base URLs given")

    owns_clients = clients is None
    if clients is None:
        clients = [AsyncOpenAI(base_url=u.rstrip("/"), api_key=api_key) for u in base_urls]
    extra_body: dict[str, Any] = {}
    if not thinking:
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        extra_body["top_k"] = 20

    frame_groups = group_by_frame(samples)
    n_frames = len(frame_groups)
    n_total = len(samples)
    print(
        f"  Frame grouping: {n_total} questions across {n_frames} frames "
        f"(avg {n_total / max(n_frames, 1):.1f} QA/frame)"
    )

    results: dict[int, dict[str, Any]] = {}
    t0 = time.perf_counter()

    if predictions_path is not None:
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
    fp = predictions_path.open("w", encoding="utf-8") if predictions_path else None
    next_flush = 0

    def flush() -> None:
        nonlocal next_flush
        if fp is None:
            return
        while next_flush in results:
            fp.write(json.dumps(results[next_flush], ensure_ascii=False) + "\n")
            fp.flush()
            next_flush += 1

    # In-flight requests per server, and how many frames may overlap per server.
    # Depth 2 lets the next frame prefill while the current one drains; deeper
    # pipelines start evicting the prefix cache.
    request_sems = [asyncio.Semaphore(concurrent_per_url) for _ in clients]
    frame_sems = [asyncio.Semaphore(pipeline_depth) for _ in clients]
    done_frames = 0
    done_qa = 0

    async def run_frame_group(group: list[tuple[int, dict[str, Any]]], client_idx: int) -> None:
        nonlocal done_frames, done_qa
        async with frame_sems[client_idx]:
            sample0 = group[0][1]
            image_parts = build_image_parts(
                sample0.get("image_paths") or [],
                sample0.get("image_contexts"),
            )

            async def one(idx: int, sample: dict[str, Any]) -> None:
                async with request_sems[client_idx]:
                    results[idx] = await run_one(
                        clients[client_idx],
                        sample,
                        model,
                        extra_body,
                        max_retries=max_retries,
                        prebuilt_image_parts=image_parts,
                    )

            try:
                await asyncio.gather(*[one(idx, s) for idx, s in group])
            finally:
                image_parts.clear()
            flush()

            done_frames += 1
            done_qa += len(group)
            if done_frames % max(1, n_frames // 20) == 0 or done_frames == n_frames:
                elapsed = time.perf_counter() - t0
                speed = done_qa / elapsed if elapsed > 0 else 0.0
                eta = (n_total - done_qa) / speed if speed > 0 else 0.0
                print(
                    f"  Progress: {done_frames}/{n_frames} frames, "
                    f"{done_qa}/{n_total} questions, {elapsed:.0f}s elapsed, "
                    f"{speed:.1f} QA/s, ETA {eta:.0f}s",
                    flush=True,
                )

    try:
        await asyncio.gather(
            *[
                run_frame_group(g, gi % len(clients))
                for gi, g in enumerate(frame_groups)
            ]
        )
        flush()
        return [results.get(i, {}) for i in range(n_total)]
    finally:
        if fp is not None:
            fp.close()
        if owns_clients:
            await asyncio.gather(
                *(client.close() for client in clients),
                return_exceptions=True,
            )
