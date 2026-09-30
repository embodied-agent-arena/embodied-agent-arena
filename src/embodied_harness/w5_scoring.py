"""Private W5 affordance scorers. Ground truth never crosses the agent RPC."""
from __future__ import annotations

from pathlib import Path
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
        if "[" in text and "]" in text:
            text = text[text.find("[") + 1:text.rfind("]")]
        parts = [p for p in text.replace(";", " ").replace(",", " ").split() if p]
        values = [_as_float(p) for p in parts]
        if any(item is None for item in values):
            return None
        value = values
    if isinstance(value, (list, tuple)) and value:
        values = [_as_float(item) for item in value]
        if any(item is None for item in values):
            return None
        if size is not None and len(values) != size:
            return None
        return [float(item) for item in values]
    return None


def parse_submission(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, (list, tuple)):
        point = _as_vector(raw, 2)
        bbox = _as_vector(raw, 4)
        if bbox is not None:
            return {"bbox_2d": bbox}
        if point is not None:
            return {"point_2d": point}
        return {}
    if not isinstance(raw, dict):
        text = str(raw).strip()
        if not text:
            return {}
        try:
            import json
            parsed = json.loads(text)
            if isinstance(parsed, (dict, list, tuple)):
                return parse_submission(parsed)
        except (ValueError, TypeError):
            pass
        bbox = _as_vector(text, 4)
        if bbox is not None:
            return {"bbox_2d": bbox}
        point = _as_vector(text, 2)
        return {"point_2d": point} if point is not None else {"text": text}
    if "answer" in raw and len(raw) == 1:
        return parse_submission(raw["answer"])
    point = _as_vector(raw.get("point_2d") or raw.get("point") or raw.get("xy"), 2)
    bbox = _as_vector(raw.get("bbox_2d") or raw.get("bbox") or raw.get("xyxy"), 4)
    result: dict[str, Any] = {}
    if point is not None:
        result["point_2d"] = point
    if bbox is not None:
        result["bbox_2d"] = bbox
    return result


def bbox_iou(pred: list[float], truth: list[float]) -> float:
    ax1, ay1, ax2, ay2 = _ordered_bbox(pred)
    bx1, by1, bx2, by2 = _ordered_bbox(truth)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom else 0.0


def _ordered_bbox(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = box
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def point_in_bbox(point: list[float], box: list[float]) -> bool:
    x, y = point
    x1, y1, x2, y2 = _ordered_bbox(box)
    return x1 <= x <= x2 and y1 <= y <= y2


def load_mask(path: Path) -> Any:
    from PIL import Image
    import numpy as np
    with Image.open(path) as img:
        return (np.array(img.convert("L")) > 0)


def point_in_mask(point: list[float], mask) -> bool:
    y = int(round(point[1]))
    x = int(round(point[0]))
    if y < 0 or x < 0 or y >= mask.shape[0] or x >= mask.shape[1]:
        return False
    return bool(mask[y, x])


def bbox_mask_iou(box: list[float], mask) -> float:
    import numpy as np
    x1, y1, x2, y2 = [int(round(v)) for v in _ordered_bbox(box)]
    height, width = mask.shape[:2]
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    pred = np.zeros_like(mask, dtype=bool)
    if x2 > x1 and y2 > y1:
        pred[y1:y2, x1:x2] = True
    inter = int((pred & mask).sum())
    union = int((pred | mask).sum())
    return inter / union if union else 0.0


def score_region(prediction: dict[str, Any], case: dict[str, Any], data_root: Path | None = None) -> dict[str, Any]:
    gt = case.get("gt") or {}
    pred_box = _as_vector(prediction.get("bbox_2d"), 4)
    pred_point = _as_vector(prediction.get("point_2d"), 2)
    truth_box = _as_vector(gt.get("bbox_2d"), 4)
    truth_point = _as_vector(gt.get("point_2d"), 2)
    mask = None
    mask_rel = gt.get("mask_path")
    if mask_rel and data_root is not None:
        mask_file = Path(data_root) / mask_rel
        if mask_file.is_file():
            mask = load_mask(mask_file)
    box_iou = bbox_iou(pred_box, truth_box) if pred_box and truth_box else 0.0
    mask_iou = bbox_mask_iou(pred_box, mask) if pred_box is not None and mask is not None else None
    inside = False
    if pred_point is not None and mask is not None:
        inside = point_in_mask(pred_point, mask)
    elif pred_point is not None and truth_box is not None:
        inside = point_in_bbox(pred_point, truth_box)
    official = mask_iou if mask_iou is not None else box_iou
    if pred_box is None and pred_point is not None:
        official = 1.0 if inside else 0.0
    passed = bool((pred_box is not None and (box_iou >= 0.5 or (mask_iou is not None and mask_iou >= 0.5))) or inside)
    return {
        "passed": passed,
        "official_score": float(official),
        "metric": "bbox_iou_0.5_or_point_in_mask",
        "bbox_iou": box_iou,
        "giou": mask_iou,
        "point_in_gt": inside,
        "has_mask": mask is not None,
    }


def score_case(prediction: dict[str, Any], case: dict[str, Any], data_root: Path | None = None) -> dict[str, Any]:
    result = score_region(prediction, case, data_root=data_root)
    result.update(benchmark=case.get("benchmark"), sample_id=case.get("sample_id"),
                  task_family=case.get("task_family"))
    return result
