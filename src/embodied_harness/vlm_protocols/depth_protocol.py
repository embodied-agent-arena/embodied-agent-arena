"""Frozen sparse monocular Z-depth protocol and metrics (no camera inputs)."""
from __future__ import annotations

import json
import math
import re

import numpy as np

PROTOCOL = "single-rgb-sparse-z-depth-v1"
POINT_COUNT = 24
PAIR_ABS_MARGIN_M = .05
PAIR_REL_MARGIN = .05
BASELINE_M = 2.0
OBJECT_COUNTS = ((1, 2), (2, 2), (3, 2), (4, 3), (6, 3))
PROMPT = """请仅根据这张 RGB 图片，估计下列采样点处可见表面的深度。
原图宽 W={width}、高 H={height} 像素。坐标以原图左上角外边缘为原点，u 向右、v 向下，左上像素中心为 (0.5,0.5)。即使图像在你内部被缩放，也必须按原图坐标定位。
深度定义：以相机光心为原点、光轴向前为 +Z，回答表面点的相机坐标 Z，单位米。不是相机到点的欧氏斜距，不是物体的尺寸，也不是物体离桌面的高度。请估计该像素实际看到的最前方表面，不要回答被遮挡表面。
未提供相机内参、位姿、物体尺寸或参照深度。单图的真实尺度存在歧义，请结合透视和常识给出你认为最合理的估计。必须分别判断各点的深度，不要假定这些点位于同一平面。
采样点列表按输出顺序排列，每项是 [u,v]：
{points}
只输出一个 JSON 对象，恰好含一个字段 depth_m；它是长度 {count} 的数组，顺序必须与采样点一致。每个元素为正数；完全无法估计的点填 null。不要解释、其他字段或 Markdown。"""


def prompt_for(record):
    width, height = record["resolution_wh"]
    points = record["points_uv"]
    if len(points) != POINT_COUNT:
        raise ValueError("Expected 24 query points")
    for u, v in points:
        if not (0 < u < width and 0 < v < height):
            raise ValueError("Query outside image")
    return PROMPT.format(width=width, height=height,
                         points=json.dumps(points, separators=(",", ":")), count=len(points))


def parse_prediction(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    def unique_object(pairs):
        if len({k for k, _ in pairs}) != len(pairs):
            raise ValueError("Duplicate JSON keys")
        return dict(pairs)
    value = json.loads(text, object_pairs_hook=unique_object)
    if not isinstance(value, dict) or set(value) != {"depth_m"}:
        raise ValueError("Expected exactly depth_m")
    values = value["depth_m"]
    if not isinstance(values, list) or len(values) != POINT_COUNT:
        raise ValueError("Expected 24 depths in query order")
    for z in values:
        if z is not None and (isinstance(z, bool) or not isinstance(z, (int, float)) or not math.isfinite(z) or z <= 0):
            raise ValueError("Depth must be finite, positive, or null")
    return {"depth_m": [None if z is None else float(z) for z in values]}


def depth_errors(prediction, truth):
    """Continuous errors are valid-point only; success/ordering include abstentions.

    The oracle log-scale alignment is a shape diagnostic, not metric accuracy.
    No shift is fitted. All scalar aggregates are subsequently image-macro means.
    """
    gt = np.asarray(truth, dtype=float)
    pred = np.asarray([np.nan if z is None else z for z in prediction], dtype=float)
    if gt.shape != pred.shape or gt.ndim != 1 or not np.all(np.isfinite(gt) & (gt > 0)):
        raise ValueError("Bad depth arrays")
    valid = np.isfinite(pred) & (pred > 0)
    result = {"points": len(gt), "valid_points": int(valid.sum()), "coverage": float(valid.mean())}
    if valid.any():
        p, g = pred[valid], gt[valid]
        residual = p - g
        log_error = np.log(p) - np.log(g)
        # exp(mean(log(gt/p))) is the optimal positive scale in log space.
        scale = float(np.exp(-np.mean(log_error)))
        ratio = np.maximum(p / g, g / p)
        result.update(abs_rel=float(np.mean(np.abs(residual) / g)),
                      mae_m=float(np.mean(np.abs(residual))), rmse_m=float(np.sqrt(np.mean(residual ** 2))),
                      delta1=float(np.sum(ratio < 1.25) / len(gt)),
                      within_10pct=float(np.sum(np.abs(residual) / g <= .10) / len(gt)),
                      oracle_scale=scale, median_pred_gt_ratio=float(np.median(p / g)),
                      scale_aligned_abs_rel=float(np.mean(np.abs(scale * p - g) / g)),
                      centered_log_rmse=float(np.sqrt(np.mean((log_error - log_error.mean()) ** 2))))
    else:
        result.update({k: None for k in ("abs_rel", "mae_m", "rmse_m", "oracle_scale",
                                        "median_pred_gt_ratio", "scale_aligned_abs_rel", "centered_log_rmse")})
        result.update(delta1=0., within_10pct=0.)
    pairs = [(a, b) for a in range(len(gt)) for b in range(a + 1, len(gt))
             if abs(gt[a] - gt[b]) > max(PAIR_ABS_MARGIN_M, PAIR_REL_MARGIN * min(gt[a], gt[b]))]
    correct = sum(bool(valid[a] and valid[b] and (pred[a] - pred[b]) * (gt[a] - gt[b]) > 0)
                  for a, b in pairs)
    result.update(ordinal_pairs=len(pairs), ordinal_correct=correct,
                  ordinal_accuracy=correct / len(pairs) if pairs else None)
    return result


def interior_mask(depth, instances, radius=4):
    """9x9 support: one instance, finite Z, and <2% depth range."""
    height, width = depth.shape
    if instances.shape != depth.shape or min(depth.shape) < 2 * radius + 1:
        raise ValueError("Invalid image shape")
    center = depth[radius:height-radius, radius:width-radius]
    ids = instances[radius:height-radius, radius:width-radius]
    valid = np.isfinite(center) & (center > 0)
    low, high = center.copy(), center.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            z = depth[radius+dy:height-radius+dy, radius+dx:width-radius+dx]
            m = instances[radius+dy:height-radius+dy, radius+dx:width-radius+dx]
            valid &= np.isfinite(z) & (z > 0) & (m == ids)
            low = np.minimum(low, z)
            high = np.maximum(high, z)
    valid &= (high - low) < .02 * center
    output = np.zeros(depth.shape, dtype=bool)
    output[radius:height-radius, radius:width-radius] = valid
    return output


def select_points(depth, instances, seed):
    """12 foreground + 12 background, with shuffled query order.

    Sampling uses masks/depth only for surface eligibility; IDs, semantic strata,
    depths and scene group are hidden from the prompt.
    """
    eligible = interior_mask(depth, instances)
    rng = np.random.default_rng(seed)
    points, strata = [], []
    for iid, count in OBJECT_COUNTS:
        candidates = np.argwhere(eligible & (instances == iid))
        if len(candidates) < count:
            raise ValueError(f"Object {iid} lacks stable interior points")
        first = candidates[int(rng.integers(len(candidates)))]
        selected = [first]
        for _ in range(count - 1):
            distance = np.min(np.sum((candidates[:, None, :] - np.asarray(selected)[None, :, :]) ** 2, axis=2), axis=1)
            if np.max(distance) < 64:
                raise ValueError(f"Object {iid} query points too close")
            selected.append(candidates[int(np.argmax(distance))])
        points.extend(selected)
        strata.extend(["object"] * count)
    candidates = np.argwhere(eligible & (instances == 0))
    if len(candidates) < 12:
        raise ValueError("Too few background points")
    height, width = depth.shape
    for row in range(3):
        for col in range(4):
            subset = candidates[(candidates[:, 0] >= row * height / 3)
                                & (candidates[:, 0] < (row + 1) * height / 3)
                                & (candidates[:, 1] >= col * width / 4)
                                & (candidates[:, 1] < (col + 1) * width / 4)]
            if not len(subset):
                subset = candidates
            distance = np.min(np.sum((subset[:, None, :] - np.asarray(points)[None, :, :]) ** 2, axis=2), axis=1)
            distant = subset[distance >= 32 ** 2]
            chosen = distant[int(rng.integers(len(distant)))] if len(distant) else subset[int(np.argmax(distance))]
            points.append(chosen)
            strata.append("environment")
    order = rng.permutation(POINT_COUNT)
    return [(int(points[i][1]), int(points[i][0]), strata[i]) for i in order]
