"""Per-question-type metrics for NuReasoning VQA evaluation."""
from __future__ import annotations

import ast
import json
import math
import re
from typing import Any

try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None  # type: ignore


def _norm_ws(s: str) -> str:
    return " ".join(s.lower().split())


def parse_choice_letter(text: str) -> str | None:
    """Extract first single letter A-D from model output (single-choice)."""
    if not text:
        return None
    m = re.search(r"\b([A-Da-d])\b", text.strip())
    return m.group(1).upper() if m else None


def parse_choice_letters_all(text: str) -> set[str]:
    """All distinct option letters A-D mentioned in the output (multi-select)."""
    return {m.group(1).upper() for m in re.finditer(r"\b([A-Da-d])\b", text or "")}


def parse_number(text: str) -> float | None:
    """Extract first numeric literal (supports simple decimals)."""
    if not text:
        return None
    cleaned = text.replace(",", "")
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cleaned)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def parse_json_or_literal(text: str) -> Any | None:
    """Parse model output as JSON or Python literal (lists for trajectories / xy)."""
    t = (text or "").strip()
    if not t:
        return None
    for fn in (json.loads, ast.literal_eval):
        try:
            return fn(t)
        except (json.JSONDecodeError, SyntaxError, ValueError, TypeError):
            continue
    return None


def parse_xy_pair(text: str) -> tuple[float, float] | None:
    """Parse [x, y] or two numbers from text."""
    obj = parse_json_or_literal(text)
    if isinstance(obj, (list, tuple)) and len(obj) >= 2:
        try:
            return float(obj[0]), float(obj[1])
        except (TypeError, ValueError):
            pass
    nums = re.findall(r"[-+]?\d*\.?\d+", (text or "").replace(",", " "))
    if len(nums) >= 2:
        try:
            return float(nums[0]), float(nums[1])
        except ValueError:
            pass
    return None


def parse_txy_trajectory(obj: Any) -> list[tuple[float, float, float]] | None:
    """[[t, x, y], ...] with at least 3 floats per row."""
    if not isinstance(obj, list) or not obj:
        return None
    out: list[tuple[float, float, float]] = []
    for row in obj:
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            return None
        try:
            out.append((float(row[0]), float(row[1]), float(row[2])))
        except (TypeError, ValueError):
            return None
    return out


def _gold_letters_from_answer(answer: Any) -> set[str] | None:
    """If gold is multi-select list of letters, return set; if not multi-select shape, None."""
    if isinstance(answer, list) and answer:
        if all(isinstance(x, str) and len(x.strip()) <= 2 for x in answer):
            return {str(x).strip().upper()[:1] for x in answer if str(x).strip()}
        return None
    return None


def score_multiselect(pred_raw: str, gold: dict[str, Any]) -> dict[str, Any]:
    """
    Multiple-choice with gold answer as list of letters, e.g. ["A", "C"].

    Metrics: exact set match, precision, recall, F1 (micro-style on this item).
    """
    gold_set = _gold_letters_from_answer(gold.get("answer"))
    if not gold_set:
        return {"multiselect_exact_match": 0.0, "error": "gold_not_multiselect_letters"}
    pred_set = parse_choice_letters_all(pred_raw)
    inter = gold_set & pred_set
    prec = len(inter) / len(pred_set) if pred_set else 0.0
    rec = len(inter) / len(gold_set) if gold_set else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    em = 1.0 if gold_set == pred_set else 0.0
    return {
        "multiselect_exact_match": em,
        "multiselect_precision": prec,
        "multiselect_recall": rec,
        "multiselect_f1": f1,
    }


def _option_text_for_letter(letter: str, choices: dict[str, Any] | None) -> str:
    if not letter or not isinstance(choices, dict):
        return ""
    return str(choices.get(letter) or choices.get(letter.upper()) or "").strip()


def score_choice(pred_raw: str, gold: dict[str, Any]) -> dict[str, Any]:
    """
    Single or multi-select (if gold answer is list of letters).

    Single: letter accuracy + relaxed text + label norms for dataset macro-F1.
    """
    ga = gold.get("answer")
    if isinstance(ga, list):
        ms = score_multiselect(pred_raw, gold)
        if "error" not in ms:
            return ms
        return {
            "choice_letter_exact": 0.0,
            "choice_text_relaxed": 0.0,
            "categorical_accuracy": 0.0,
            "choice_gold_label_norm": None,
            "choice_pred_label_norm": None,
            "note": ms.get("error", "unsupported_choice_answer_shape"),
        }

    gold_letter = str(ga).strip().upper()[:1] if ga is not None else ""
    gold_text = str(gold.get("answer_text", "")).strip()
    pred_letter = parse_choice_letter(pred_raw) or ""
    pred_norm = _norm_ws(pred_raw)
    gold_text_norm = _norm_ws(gold_text)
    choices = gold.get("choices")

    letter_ok = bool(pred_letter and gold_letter and pred_letter == gold_letter)
    text_ok = bool(gold_text_norm and gold_text_norm in pred_norm)
    text_ok = text_ok or (
        bool(gold_text_norm) and pred_norm.startswith(gold_text_norm[: min(8, len(gold_text_norm))])
    )

    pred_label_norm = _norm_ws(_option_text_for_letter(pred_letter, choices))
    gold_label_norm = gold_text_norm or _norm_ws(_option_text_for_letter(gold_letter, choices))

    return {
        "choice_letter_exact": 1.0 if letter_ok else 0.0,
        "choice_text_relaxed": 1.0 if (letter_ok or text_ok) else 0.0,
        "categorical_accuracy": 1.0 if (letter_ok or (gold_label_norm and pred_label_norm == gold_label_norm)) else 0.0,
        "parsed_letter": pred_letter or None,
        "gold_letter": gold_letter or None,
        "choice_gold_label_norm": gold_label_norm or None,
        "choice_pred_label_norm": pred_label_norm or None,
    }


def score_coordinate(
    pred_raw: str,
    gold: dict[str, Any],
    *,
    tol: float | None = None,
) -> dict[str, Any]:
    """Gold answer is [x, y] (meters). L2 distance and hit@tolerance."""
    g = gold.get("answer")
    if not isinstance(g, (list, tuple)) or len(g) < 2:
        return {"coordinate_l2": None, "coordinate_hit_at_tolerance": 0.0, "error": "bad_gold_xy"}
    try:
        gx, gy = float(g[0]), float(g[1])
    except (TypeError, ValueError):
        return {"coordinate_l2": None, "coordinate_hit_at_tolerance": 0.0, "error": "bad_gold_xy"}
    t = tol if tol is not None else float(gold.get("tolerance") or 1.0)
    pair = parse_xy_pair(pred_raw)
    if pair is None:
        return {
            "coordinate_l2": None,
            "coordinate_hit_at_tolerance": 0.0,
            "parsed_xy": None,
        }
    px, py = pair
    l2 = math.hypot(px - gx, py - gy)
    return {
        "coordinate_l2": l2,
        "coordinate_hit_at_tolerance": 1.0 if l2 <= t else 0.0,
        "parsed_xy": [px, py],
        "gold_xy": [gx, gy],
    }


def score_trajectory(pred_raw: str, gold: dict[str, Any]) -> dict[str, Any]:
    """
    Gold answer [[t, x, y], ...]. Per-waypoint L2 in (x,y), mean, RMSE, hit fraction.
    """
    g = gold.get("answer")
    traj_g = parse_txy_trajectory(g) if isinstance(g, list) else None
    if not traj_g:
        return {"trajectory_mean_l2_xy": None, "error": "bad_gold_trajectory"}
    pred_obj = parse_json_or_literal(pred_raw)
    traj_p = parse_txy_trajectory(pred_obj) if pred_obj is not None else None
    tol = float(gold.get("tolerance") or 1.0)
    if not traj_p:
        return {
            "trajectory_mean_l2_xy": None,
            "trajectory_rmse_xy": None,
            "trajectory_hit_at_tolerance": 0.0,
        }
    n = min(len(traj_p), len(traj_g))
    if n == 0:
        return {
            "trajectory_mean_l2_xy": None,
            "trajectory_rmse_xy": None,
            "trajectory_hit_at_tolerance": 0.0,
        }
    l2s: list[float] = []
    hits = 0
    for i in range(n):
        dx = traj_p[i][1] - traj_g[i][1]
        dy = traj_p[i][2] - traj_g[i][2]
        d = math.hypot(dx, dy)
        l2s.append(d)
        if d <= tol:
            hits += 1
    mean_l2 = sum(l2s) / n
    rmse = math.sqrt(sum(d * d for d in l2s) / n)
    hit_frac = hits / n
    return {
        "trajectory_mean_l2_xy": mean_l2,
        "trajectory_rmse_xy": rmse,
        "trajectory_hit_at_tolerance": hit_frac,
        "trajectory_waypoints_compared": float(n),
    }


def score_numerical(
    pred_raw: str,
    gold: dict[str, Any],
    *,
    rtol: float = 0.05,
    atol: float = 0.5,
) -> dict[str, Any]:
    """
    Scalar: tolerance accuracy + errors.
    List: trajectory (txy) or 2-vector coordinate.
    """
    g = gold.get("answer")
    if isinstance(g, list) and g:
        fmt = str(gold.get("answer_format") or "")
        if fmt == "txy_sequence_m" or (isinstance(g[0], (list, tuple)) and len(g[0]) >= 3):
            return score_trajectory(pred_raw, gold)
        if len(g) == 2:
            try:
                float(g[0])
                float(g[1])
                return score_coordinate(pred_raw, gold)
            except (TypeError, ValueError):
                pass

    if g is None:
        return {"numerical_within_tolerance": 0.0, "error": "missing_gold"}
    try:
        gold_v = float(g)
    except (TypeError, ValueError):
        return {"numerical_within_tolerance": 0.0, "error": "bad_gold"}

    pred_v = parse_number(pred_raw)
    if pred_v is None:
        return {
            "numerical_within_tolerance": 0.0,
            "numerical_abs_error": None,
            "numerical_relative_error": None,
            "parsed_value": None,
        }

    use_atol = float(gold.get("tolerance")) if gold.get("tolerance") is not None else atol
    diff = abs(pred_v - gold_v)
    tol = use_atol + rtol * max(abs(gold_v), 1e-9)
    ok = diff <= tol
    rel = diff / max(abs(gold_v), 1e-9)

    return {
        "numerical_within_tolerance": 1.0 if ok else 0.0,
        "numerical_abs_error": diff,
        "numerical_relative_error": rel,
        "parsed_value": pred_v,
        "gold_value": gold_v,
    }


def _rouge_l_f1(reference: str, hypothesis: str) -> float | None:
    if not reference.strip() or not hypothesis.strip():
        return 0.0
    if rouge_scorer is None:
        return None
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return float(scorer.score(reference, hypothesis)["rougeL"].fmeasure)


def _token_f1(reference: str, hypothesis: str) -> float:
    rt = _norm_ws(reference).split()
    ht = _norm_ws(hypothesis).split()
    if not rt or not ht:
        return 0.0
    rs, hs = set(rt), set(ht)
    inter = len(rs & hs)
    if inter == 0:
        return 0.0
    prec = inter / len(hs)
    rec = inter / len(rs)
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def score_text(pred_raw: str, gold: dict[str, Any]) -> dict[str, Any]:
    """
    Free-text. Includes categorical-style exact match on normalized string.
    """
    ref = str(gold.get("answer_text") or gold.get("answer") or "").strip()
    hyp = pred_raw.strip()
    exact = 1.0 if _norm_ws(ref) == _norm_ws(hyp) and ref else 0.0
    rouge = _rouge_l_f1(ref, hyp)
    tok_f1 = _token_f1(ref, hyp)
    out: dict[str, Any] = {
        "text_exact_normalized": exact,
        "text_token_f1": tok_f1,
        "categorical_accuracy": exact,
        "text_gold_label_norm": _norm_ws(ref) if ref else None,
        "text_pred_label_norm": _norm_ws(hyp) if hyp else None,
    }
    if rouge is not None:
        out["text_rougeL_f1"] = rouge
    return out


def score_categorical(question_type: str, pred_raw: str, gold: dict[str, Any]) -> dict[str, Any]:
    """Explicit categorical type: match normalized answer_text."""
    ref = str(gold.get("answer_text") or gold.get("answer") or "").strip()
    hyp = pred_raw.strip()
    exact = 1.0 if _norm_ws(ref) == _norm_ws(hyp) and ref else 0.0
    return {
        "categorical_accuracy": exact,
        "categorical_gold_label_norm": _norm_ws(ref) if ref else None,
        "categorical_pred_label_norm": _norm_ws(hyp) if hyp else None,
    }


def compute_metrics_for_sample(question_type: str, pred_raw: str, gold_question: dict[str, Any]) -> dict[str, Any]:
    """Dispatch to type-specific metrics."""
    qt = (question_type or "").lower()
    if qt == "choice":
        return score_choice(pred_raw, gold_question)
    if qt == "numerical":
        return score_numerical(pred_raw, gold_question)
    if qt == "text":
        return score_text(pred_raw, gold_question)
    if qt == "categorical":
        return score_categorical(qt, pred_raw, gold_question)
    return {"unknown_type": 1.0, "note": f"no metrics for question_type={question_type}"}


def primary_metric_name(question_type: str) -> str:
    qt = (question_type or "").lower()
    if qt == "choice":
        return "choice_letter_exact"
    if qt == "numerical":
        return "numerical_within_tolerance"
    if qt == "text":
        return "text_rougeL_f1"
    if qt == "categorical":
        return "categorical_accuracy"
    return "unknown_type"


def primary_metric_value(metrics: dict[str, Any], question_type: str) -> float | None:
    qt = (question_type or "").lower()
    if qt == "numerical" and metrics.get("trajectory_hit_at_tolerance") is not None:
        v = metrics.get("trajectory_hit_at_tolerance")
        return float(v) if v is not None else None
    if qt == "numerical" and metrics.get("coordinate_l2") is not None:
        v = metrics.get("coordinate_hit_at_tolerance")
        return float(v) if v is not None else None
    if qt == "choice" and "multiselect_f1" in metrics:
        return float(metrics["multiselect_f1"])

    name = primary_metric_name(question_type)
    if name in metrics:
        v = metrics[name]
        return float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else None
    if qt == "text" and "text_token_f1" in metrics:
        return float(metrics["text_token_f1"])
    return None


def _macro_f1_multiclass(labels_true: list[str], labels_pred: list[str]) -> float | None:
    """Macro F1 over classes present in labels_true (string labels)."""
    if not labels_true or len(labels_true) != len(labels_pred):
        return None
    classes = sorted(set(labels_true))
    if not classes:
        return None
    f1s: list[float] = []
    for c in classes:
        tp = fp = fn = 0
        for t, p in zip(labels_true, labels_pred):
            if t == c and p == c:
                tp += 1
            elif t != c and p == c:
                fp += 1
            elif t == c and p != c:
                fn += 1
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s)


def compute_dataset_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Dataset-level metrics: MAE/RMSE for scalars, macro-F1 on choice/text labels, trajectory means.
    """
    out: dict[str, Any] = {}

    # Choice: macro-F1 and accuracy on normalized answer_text labels
    y_true_l: list[str] = []
    y_pred_l: list[str] = []
    for r in rows:
        if r.get("error"):
            continue
        if str(r.get("question_type") or "").lower() != "choice":
            continue
        m = r.get("metrics") or {}
        tg = m.get("choice_gold_label_norm")
        pr = m.get("choice_pred_label_norm")
        if isinstance(tg, str) and tg:
            y_true_l.append(tg)
            y_pred_l.append(pr if isinstance(pr, str) and pr else "")
    if y_true_l:
        mf = _macro_f1_multiclass(y_true_l, y_pred_l)
        if mf is not None:
            out["choice_macro_f1_answer_text"] = mf
        acc = sum(1 for t, p in zip(y_true_l, y_pred_l) if t == p) / len(y_true_l)
        out["choice_accuracy_answer_text"] = acc

    # Text / categorical macro-F1 on normalized free-text labels
    y_tt: list[str] = []
    y_pt: list[str] = []
    for r in rows:
        if r.get("error"):
            continue
        qt = str(r.get("question_type") or "").lower()
        if qt not in ("text", "categorical"):
            continue
        m = r.get("metrics") or {}
        tg = m.get("text_gold_label_norm") or m.get("categorical_gold_label_norm")
        pr = m.get("text_pred_label_norm") or m.get("categorical_pred_label_norm")
        if isinstance(tg, str) and tg:
            y_tt.append(tg)
            y_pt.append(pr if isinstance(pr, str) and pr else "")
    if y_tt:
        mf = _macro_f1_multiclass(y_tt, y_pt)
        if mf is not None:
            out["text_macro_f1_answer_text"] = mf
        out["text_accuracy_answer_text"] = sum(1 for t, p in zip(y_tt, y_pt) if t == p) / len(y_tt)

    # Numerical scalar: MAE, RMSE, mean within tolerance
    errs: list[float] = []
    tol_ok: list[float] = []
    for r in rows:
        if r.get("error"):
            continue
        if str(r.get("question_type") or "").lower() != "numerical":
            continue
        m = r.get("metrics") or {}
        ae = m.get("numerical_abs_error")
        if isinstance(ae, (int, float)):
            errs.append(float(ae))
        wt = m.get("numerical_within_tolerance")
        if isinstance(wt, (int, float)):
            tol_ok.append(float(wt))
    if errs:
        out["numerical_scalar_mae"] = sum(errs) / len(errs)
        out["numerical_scalar_rmse"] = math.sqrt(sum(e * e for e in errs) / len(errs))
    if tol_ok:
        out["numerical_scalar_mean_accuracy_within_tolerance"] = sum(tol_ok) / len(tol_ok)

    # Trajectory aggregates
    l2s: list[float] = []
    rmses: list[float] = []
    hits: list[float] = []
    for r in rows:
        if r.get("error"):
            continue
        m = r.get("metrics") or {}
        if m.get("trajectory_mean_l2_xy") is not None:
            l2s.append(float(m["trajectory_mean_l2_xy"]))
        if m.get("trajectory_rmse_xy") is not None:
            rmses.append(float(m["trajectory_rmse_xy"]))
        if m.get("trajectory_hit_at_tolerance") is not None:
            hits.append(float(m["trajectory_hit_at_tolerance"]))
    if l2s:
        out["trajectory_mean_of_mean_l2_xy"] = sum(l2s) / len(l2s)
    if rmses:
        out["trajectory_mean_of_rmse_xy"] = sum(rmses) / len(rmses)
    if hits:
        out["trajectory_mean_hit_at_tolerance"] = sum(hits) / len(hits)

    # Coordinate aggregates
    cl2: list[float] = []
    chit: list[float] = []
    for r in rows:
        if r.get("error"):
            continue
        m = r.get("metrics") or {}
        if m.get("coordinate_l2") is not None:
            cl2.append(float(m["coordinate_l2"]))
        if m.get("coordinate_hit_at_tolerance") is not None:
            chit.append(float(m["coordinate_hit_at_tolerance"]))
    if cl2:
        out["coordinate_mean_l2"] = sum(cl2) / len(cl2)
        out["coordinate_rmse_l2"] = math.sqrt(sum(x * x for x in cl2) / len(cl2))
    if chit:
        out["coordinate_mean_hit_at_tolerance"] = sum(chit) / len(chit)

    # Multi-select (if any)
    mf1s = [
        float(r["metrics"]["multiselect_f1"])
        for r in rows
        if not r.get("error")
        and r.get("metrics", {}).get("multiselect_f1") is not None
    ]
    if mf1s:
        out["multiselect_mean_f1"] = sum(mf1s) / len(mf1s)
        ems = [
            float(r["metrics"]["multiselect_exact_match"])
            for r in rows
            if not r.get("error")
            and r.get("metrics", {}).get("multiselect_exact_match") is not None
        ]
        if ems:
            out["multiselect_mean_exact_match"] = sum(ems) / len(ems)

    return out
