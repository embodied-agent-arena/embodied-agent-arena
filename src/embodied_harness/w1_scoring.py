"""Private W1 geometry scorers. Ground truth never crosses the agent RPC."""
from __future__ import annotations

import math
import re
from typing import Any


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _as_vector(value: Any, size: int | None = None) -> list[float] | None:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("`") and text.endswith("`"):
            text = text[1:-1]
        match = re.search(r"\[([^\[\]]+)\]", text)
        if match:
            text = match.group(1)
        parts = [p for p in re.split(r"[\s,;]+", text) if p]
        values = [_as_float(p) for p in parts]
        if any(v is None for v in values):
            return None
        value = values
    if isinstance(value, (list, tuple)) and value:
        values = [_as_float(v) for v in value]
        if any(v is None for v in values):
            return None
        if size is not None and len(values) != size:
            return None
        return [float(v) for v in values]
    return None


def parse_submission(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return {"value": float(raw)}
    if isinstance(raw, (list, tuple)):
        vector = _as_vector(raw)
        return {"vector": vector} if vector is not None else {}
    if not isinstance(raw, dict):
        text = str(raw).strip()
        if not text:
            return {}
        try:
            import json
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parse_submission(parsed)
            if isinstance(parsed, (list, tuple, int, float)):
                return parse_submission(parsed)
        except (ValueError, TypeError):
            pass
        vector = _as_vector(text)
        if vector is not None:
            return {"vector": vector} if len(vector) > 1 else {"value": vector[0]}
        number = _as_float(text)
        if number is not None:
            return {"value": number}
        return {"text": text}
    result = dict(raw)
    if "answer" in result and len(result) == 1:
        return parse_submission(result["answer"])
    return result


def _norm(values: list[float]) -> float:
    return math.sqrt(sum(v * v for v in values))


def score_multispa(prediction: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    answer_type = case.get("answer_type") or "scalar"
    gt = case.get("gt") or {}
    width = float((case.get("images") or [{}])[0].get("width") or 0)
    if answer_type in {"choice", "qualitative", "text"}:
        pred = str(prediction.get("choice") or prediction.get("text") or prediction.get("value") or "").strip()
        truth = str(gt.get("label") or gt.get("value") or "").strip()
        passed = bool(pred) and pred.casefold() == truth.casefold()
        return {"passed": passed, "official_score": 1.0 if passed else 0.0, "metric": "exact_match"}
    if answer_type in {"pixel", "coord", "coordinates"}:
        pred = _as_vector(prediction.get("vector") or prediction.get("value") or prediction.get("xy"), 2)
        truth = _as_vector(gt.get("xy") or gt.get("vector") or gt.get("value"), 2)
        if pred is None or truth is None or width <= 0:
            return {"passed": False, "official_score": 0.0, "metric": "pixel_5pct_width"}
        distance = _norm([pred[0] - truth[0], pred[1] - truth[1]])
        passed = distance <= 0.05 * width
        return {
            "passed": passed,
            "official_score": 1.0 if passed else 0.0,
            "metric": "pixel_5pct_width",
            "pixel_error": distance,
        }
    pred = _as_vector(prediction.get("vector") or prediction.get("value") or prediction.get("displacement"))
    truth = _as_vector(gt.get("vector") or gt.get("value") or gt.get("displacement"))
    if pred is None:
        scalar = _as_float(prediction.get("value") or prediction.get("scalar"))
        pred = None if scalar is None else [scalar]
    if truth is None:
        scalar = _as_float(gt.get("value") or gt.get("scalar"))
        truth = None if scalar is None else [scalar]
    if pred is None or truth is None or len(pred) != len(truth):
        return {"passed": False, "official_score": 0.0, "metric": "relative_l2_20pct"}
    error = _norm([p - t for p, t in zip(pred, truth)])
    scale = _norm(truth)
    passed = error <= 0.2 * scale if scale > 0 else error == 0
    return {
        "passed": passed,
        "official_score": 1.0 if passed else 0.0,
        "metric": "relative_l2_20pct",
        "l2_error": error,
        "relative_l2": (error / scale) if scale else None,
    }


def score_influx(prediction: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    gt = (case.get("gt") or {}).get("intrinsics") or case.get("gt") or {}
    pred_k = prediction.get("intrinsics") if isinstance(prediction.get("intrinsics"), dict) else prediction
    keys = ("fx", "fy", "cx", "cy")
    pred = {k: _as_float(pred_k.get(k)) for k in keys}
    truth = {k: _as_float(gt.get(k)) for k in keys}
    invalid_gt = [k for k in keys if truth[k] is None or not math.isfinite(truth[k])
                  or (k in ("fx", "fy") and truth[k] <= 0)]
    if invalid_gt:
        return {"passed": None, "official_score": None, "metric": "k_mape_pp",
                "evaluation_valid": False, "invalid_reason": "invalid_ground_truth",
                "invalid_gt_fields": invalid_gt}
    invalid_pred = [k for k in keys if pred[k] is None or not math.isfinite(pred[k])
                    or (k in ("fx", "fy") and pred[k] <= 0)]
    if invalid_pred:
        return {"passed": False, "official_score": 0.0, "metric": "k_mape_pp",
                "invalid_reason": "invalid_prediction", "invalid_prediction_fields": invalid_pred}
    if any(pred[k] is None or truth[k] is None or truth[k] == 0 for k in ("fx", "fy")):
        return {"passed": False, "official_score": 0.0, "metric": "k_mape_pp"}
    width = float((case.get("images") or [{}])[0].get("width") or 0)
    height = float((case.get("images") or [{}])[0].get("height") or 0)
    diagonal = math.hypot(width, height) if width and height else 0.0
    fx_rel = abs(pred["fx"] - truth["fx"]) / abs(truth["fx"])
    fy_rel = abs(pred["fy"] - truth["fy"]) / abs(truth["fy"])
    focal_mape = 0.5 * (fx_rel + fy_rel)
    pp_error = None
    pp_rel = None
    if pred["cx"] is not None and pred["cy"] is not None and truth["cx"] is not None and truth["cy"] is not None:
        pp_error = math.hypot(pred["cx"] - truth["cx"], pred["cy"] - truth["cy"])
        pp_rel = (pp_error / diagonal) if diagonal else None
    passed = fx_rel <= 0.10 and fy_rel <= 0.10 and (pp_rel is None or pp_rel <= 0.02)
    return {
        "passed": passed,
        "official_score": 1.0 if passed else 0.0,
        "metric": "k_10pct_pp_2pct_diag",
        "focal_mape": focal_mape,
        "fx_rel": fx_rel,
        "fy_rel": fy_rel,
        "principal_point_error_px": pp_error,
        "principal_point_rel_diag": pp_rel,
    }


def _quat_to_matrix(qw: float, qx: float, qy: float, qz: float) -> list[list[float]]:
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n <= 0:
        raise ValueError("zero quaternion")
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return [
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ]


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [list(row) for row in zip(*matrix)]


def geodesic_rotation_deg(pred_q: list[float], gt_q: list[float]) -> float:
    pred = _quat_to_matrix(*pred_q)
    truth = _quat_to_matrix(*gt_q)
    relative = _matmul(_transpose(truth), pred)
    trace = relative[0][0] + relative[1][1] + relative[2][2]
    cos = max(-1.0, min(1.0, (trace - 1.0) * 0.5))
    return math.degrees(math.acos(cos))


def _pose_from(prediction: dict[str, Any]) -> tuple[list[float] | None, list[float] | None]:
    quat = _as_vector(
        prediction.get("q")
        or prediction.get("quaternion")
        or [prediction.get(k) for k in ("qw", "qx", "qy", "qz")],
        4,
    )
    if quat is None:
        quat = _as_vector([prediction.get(k) for k in ("qw", "qx", "qy", "qz")], 4)
    trans = _as_vector(
        prediction.get("t")
        or prediction.get("translation")
        or [prediction.get(k) for k in ("tx", "ty", "tz")],
        3,
    )
    if trans is None:
        trans = _as_vector([prediction.get(k) for k in ("tx", "ty", "tz")], 3)
    return quat, trans


def score_mapfree(prediction: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    gt = case.get("gt") or {}
    pred_q, pred_t = _pose_from(prediction)
    gt_q, gt_t = _pose_from(gt)
    if pred_q is None or pred_t is None or gt_q is None or gt_t is None:
        return {"passed": False, "official_score": 0.0, "metric": "pose_2deg_2cm"}
    rot = geodesic_rotation_deg(pred_q, gt_q)
    trans = _norm([p - t for p, t in zip(pred_t, gt_t)])
    trans_cm = trans * 100.0
    direction = None
    if _norm(gt_t) > 1e-6 and _norm(pred_t) > 1e-6:
        dot = sum(p * t for p, t in zip(pred_t, gt_t)) / (_norm(pred_t) * _norm(gt_t))
        direction = math.degrees(math.acos(max(-1.0, min(1.0, dot))))
    elif _norm(gt_t) > 1e-6 and _norm(pred_t) <= 1e-6:
        direction = 180.0
    passed = rot <= 2.0 and trans_cm <= 2.0
    return {
        "passed": passed,
        "official_score": 1.0 if passed else 0.0,
        "metric": "pose_2deg_2cm",
        "rotation_error_deg": rot,
        "translation_error_m": trans,
        "translation_error_cm": trans_cm,
        "translation_direction_error_deg": direction,
    }


SCORERS = {
    "multispa": score_multispa,
    "influx": score_influx,
    "mapfree": score_mapfree,
}


def score_case(prediction: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    benchmark = case.get("benchmark")
    if isinstance(benchmark, str) and benchmark.startswith("vlm_"):
        from .w1_vlm_scoring import score
        return score(prediction, case)
    scorer = SCORERS.get(benchmark)
    if scorer is None:
        raise ValueError(f"Unknown W1 benchmark: {benchmark}")
    result = scorer(prediction, case)
    result.update(benchmark=benchmark, sample_id=case.get("sample_id"), task_family=case.get("task_family"))
    return result
