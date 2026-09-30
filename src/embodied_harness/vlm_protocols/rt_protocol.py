"""Relative camera-pose benchmark: fixed conventions and predeclared metrics."""
from __future__ import annotations

import json
import math
import re

import numpy as np

PROTOCOL = "two-view-camera-pose-unknown-intrinsics-scale-v1"
PROMPT = """这两张图片依次为同一个静止场景的图 A、图 B。所有物体与光照保持不变，只有相机可能移动或转动，也可能完全不动。同一对图片使用同一相机，内参保持不变、没有变焦。没有提供相机内参、物体真实尺寸或任何米制标定信息。

请仅根据这两张图片，估计相机 B 相对于相机 A 的三维位置和朝向。
坐标系：以相机 A 的光心为原点，+X 向画面右方、+Y 向画面下方、+Z 向相机 A 前方（看向场景），为右手坐标系。
translation_cm = [tx, ty, tz]：相机 B 光心在上述 A 坐标系中的坐标，单位厘米。这是相机本身的位移，不是画面中物体的表观运动，也不是把 A 中的点变换到 B 的外参 tvec。
rotation_xyz_deg = [rx, ry, rz]：相机 B 相对于 A 的姿态，绕 A 的固定 X、Y、Z 轴依次按右手定则旋转，单位度。明确地说，R_A_from_B = Rz(rz) @ Ry(ry) @ Rx(rx)，且 p_A = R_A_from_B @ p_B + translation_cm（p 使用厘米）。因此不动时六个数均为 0。

在没有真实尺寸的情况下，厘米位移依赖视觉尺度先验，并非唯一可恢复的量；仍请给出你最合理的数值估计。不要假定所有变化都来自平移或都来自旋转。
只输出以下 JSON 对象，两个数组各含 3 个有限数字，不要输出说明或 Markdown：
{"translation_cm": [tx, ty, tz], "rotation_xyz_deg": [rx, ry, rz]}"""

THRESHOLDS = {"rotation_deg": 2., "translation_vector_cm": 2.,
              "translation_direction_deg": [15., 30.], "magnitude_error_cm": 2.,
              "magnitude_relative_pct": 20., "zero_vector_norm_cm": 1e-6}


def rotation_matrix(angles_deg):
    """Active extrinsic XYZ: B axes expressed in the fixed A frame."""
    x, y, z = np.deg2rad(angles_deg)
    cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    rx = np.array([[1., 0., 0.], [0., cx, -sx], [0., sx, cx]])
    ry = np.array([[cy, 0., sy], [0., 1., 0.], [-sy, 0., cy]])
    rz = np.array([[cz, -sz, 0.], [sz, cz, 0.], [0., 0., 1.]])
    return rz @ ry @ rx


def pose_matrix(prediction):
    """T_A_from_B with translation converted from centimetres to metres."""
    result = np.eye(4)
    result[:3, :3] = rotation_matrix(prediction["rotation_xyz_deg"])
    result[:3, 3] = np.asarray(prediction["translation_cm"], dtype=float) / 100.
    return result


def rotation_error_deg(predicted, actual):
    cosine = (np.trace(predicted.T @ actual) - 1.) / 2.
    return math.degrees(math.acos(float(np.clip(cosine, -1., 1.))))


def parse_prediction(text):
    text = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    value = json.loads(text)
    fields = {"translation_cm", "rotation_xyz_deg"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("Expected exactly translation_cm and rotation_xyz_deg")
    for field in fields:
        vector = value[field]
        if (not isinstance(vector, list) or len(vector) != 3
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in vector)):
            raise ValueError(f"Expected three finite numbers for {field}")
    return {k: [float(v) for v in value[k]] for k in ("translation_cm", "rotation_xyz_deg")}


def errors(prediction, truth):
    pred_t, true_t = np.asarray(prediction["translation_cm"]), np.asarray(truth["translation_cm"])
    pred_norm, true_norm = float(np.linalg.norm(pred_t)), float(np.linalg.norm(true_t))
    moved = true_norm > THRESHOLDS["zero_vector_norm_cm"]
    predicted_direction = pred_norm > THRESHOLDS["zero_vector_norm_cm"]
    direction = None
    if moved and predicted_direction:
        direction = math.degrees(math.acos(float(np.clip(np.dot(pred_t / pred_norm, true_t / true_norm), -1., 1.))))
    rot = rotation_error_deg(rotation_matrix(prediction["rotation_xyz_deg"]), rotation_matrix(truth["rotation_xyz_deg"]))
    true_rot = rotation_error_deg(np.eye(3), rotation_matrix(truth["rotation_xyz_deg"]))
    t_error = float(np.linalg.norm(pred_t - true_t))
    magnitude_error = abs(pred_norm - true_norm)
    return {"rotation_error_deg": rot, "true_rotation_deg": true_rot,
            "predicted_rotation_deg": rotation_error_deg(np.eye(3), rotation_matrix(prediction["rotation_xyz_deg"])),
            "translation_vector_error_cm": t_error,
            "true_translation_norm_cm": true_norm, "predicted_translation_norm_cm": pred_norm,
            "translation_magnitude_error_cm": magnitude_error,
            "translation_magnitude_ape_pct": magnitude_error / true_norm * 100. if moved else None,
            "translation_direction_error_deg": direction,
            # A zero estimate on a moving pair is a miss; never drop it from the denominator.
            "direction_penalized_error_deg": (direction if direction is not None else 180.) if moved else None,
            "direction_defined": moved and predicted_direction,
            "translation_is_nonzero": moved, "rotation_is_nonzero": true_rot > 1e-4,
            "rotation_within_2deg": rot <= THRESHOLDS["rotation_deg"] + 1e-8,
            "translation_within_2cm": t_error <= THRESHOLDS["translation_vector_cm"] + 1e-8,
            "joint_within_2deg_2cm": rot <= THRESHOLDS["rotation_deg"] + 1e-8 and t_error <= THRESHOLDS["translation_vector_cm"] + 1e-8,
            "direction_within_15deg": moved and direction is not None and direction <= 15. + 1e-8,
            "direction_within_30deg": moved and direction is not None and direction <= 30. + 1e-8,
            "magnitude_within_2cm": moved and magnitude_error <= THRESHOLDS["magnitude_error_cm"] + 1e-8,
            "magnitude_within_20pct": moved and magnitude_error / true_norm <= .2 + 1e-8}


def zero_prediction():
    return {"translation_cm": [0., 0., 0.], "rotation_xyz_deg": [0., 0., 0.]}
