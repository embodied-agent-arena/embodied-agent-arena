"""Private native metrics for vlm_test. Continuous tasks do not invent pass gates."""
import json
import math
import statistics as stats

from .vlm_protocols import motion_protocol as motion
from .vlm_protocols import intrinsics_protocol as intrinsics
from .vlm_protocols import rt_protocol as pose
from .vlm_protocols import depth_protocol as depth

BENCHMARKS = {
    "vlm_size": "VLM desktop size",
    "vlm_motion_fixed": "VLM fixed-camera object motion",
    "vlm_motion_camera": "VLM changing-camera object motion (small)",
    "vlm_motion_camera_large": "VLM changing-camera object motion (large)",
    "vlm_motion_real": "VLM real-photo object motion",
    "vlm_intrinsics": "VLM synthetic intrinsics",
    "vlm_focal": "VLM centered focal length",
    "vlm_pose": "VLM relative camera pose",
    "vlm_depth": "VLM sparse Z depth",
    "vlm_reference": "VLM controlled reference QA",
}


def positive_fields(pred, fields):
    if set(pred) != set(fields):
        raise ValueError("Answer fields do not match the task schema")
    if any(isinstance(pred[k], bool) or not isinstance(pred[k], (int, float))
           or not math.isfinite(pred[k]) or pred[k] <= 0 for k in fields):
        raise ValueError("Expected finite positive numbers")
    return pred


def score(prediction, case):
    b, gt = case["benchmark"], case["gt"]
    base = dict(benchmark=b, sample_id=case["sample_id"], task_family=case["task_family"],
                metric=b, passed=None, official_score=None, score_kind="local_diagnostic",
                submission_valid=True, primary_metric=case["primary_metric"])
    try:
        raw = json.dumps(prediction, allow_nan=False)
        if b.startswith("vlm_motion_"):
            p = motion.parse_prediction(raw)
            error = abs(p["distance_cm"] - gt["distance_cm"])
            return dict(base, movement_correct=p["moved"] == gt["moved"],
                        direction_correct=p["direction"] == gt["direction"],
                        moving=gt["moved"], absolute_error_cm=error,
                        distance_ape_pct=100 * error / gt["distance_cm"] if gt["moved"] else None,
                        stationary_false_positive=not gt["moved"] and p["moved"],
                        joint_direction_and_1cm=p["direction"] == gt["direction"] and error <= 1.0001)
        if b == "vlm_intrinsics":
            p = intrinsics.parse_prediction(raw)
            m = intrinsics.errors(p, gt, case["images"][0]["width"], case["images"][0]["height"])
            return dict(base, **m, passed=m["joint_within_10pct_focal_2pct_diagonal"])
        if b == "vlm_focal":
            p = intrinsics.parse_prediction(raw, intrinsics.FOCAL_PROTOCOL)
            rel = p["f"] / gt["f"] - 1
            return dict(base, focal_ape_pct=abs(rel)*100, focal_signed_error_pct=rel*100,
                        focal_absolute_error_px=abs(p["f"]-gt["f"]), passed=abs(rel) <= .10)
        if b == "vlm_pose":
            m = pose.errors(pose.parse_prediction(raw), gt)
            return dict(base, **m, passed=m["joint_within_2deg_2cm"])
        if b == "vlm_depth":
            p = depth.parse_prediction(raw)["depth_m"]
            return dict(base, **depth.depth_errors(p, gt["depth_m"]))
        if b in {"vlm_size", "vlm_reference"}:
            if any(isinstance(v, str) for v in gt.values()):
                if set(prediction) != set(gt) or any(not isinstance(v, str) for v in prediction.values()):
                    raise ValueError("Expected the categorical answer schema")
                return dict(base, passed=prediction == gt, exact_match=prediction == gt)
            p = positive_fields(prediction, gt)
            absolute = {k: abs(p[k]-gt[k]) for k in gt}
            relative = {k: absolute[k]/gt[k] for k in gt}
            return dict(base, absolute_errors=absolute, relative_errors=relative,
                        mae=stats.mean(absolute.values()), mape_pct=100*stats.mean(relative.values()),
                        within_10pct_count=sum(v <= .10+1e-12 for v in relative.values()),
                        measurement_count=len(gt))
        raise ValueError("Unknown VLM benchmark")
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        return dict(base, passed=False, submission_valid=False,
                    invalid_reason="invalid_prediction", invalid_detail=str(exc))
