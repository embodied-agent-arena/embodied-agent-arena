"""Shared single-image camera-intrinsics prompt, parsing, and scoring."""
from __future__ import annotations

import json
import math
import re

PROTOCOL = "single-image-pinhole-intrinsics-v1"
FOCAL_PROTOCOL = "single-image-centered-focal-v1"
FIELDS = ("fx", "fy", "cx", "cy")
PROMPT = """请仅根据这张图片估计拍摄它的相机内参。
原始图片宽度 W={width} 像素，高度 H={height} 像素。所有答案必须使用这个原始分辨率的像素单位，即使你的视觉输入被内部缩放。
已知这是针孔透视相机，方形像素、零 skew、无镜头畸变。因此 fx=fy；主点可能偏离图像中心，请根据图片判断，不能默认居中。
坐标原点是图像左上角的外边缘，u 向右、v 向下，左上角像素中心为 (0.5,0.5)。内参矩阵 K=[[fx,0,cx],[0,fy,cy],[0,0,1]]。
fx、fy 是像素焦距，不是毫米；cx、cy 是主点坐标，不是画面中物体的中心。
没有提供传感器大小、拍摄距离、物体尺寸、相机位姿或标定板。请结合透视线索，给出你认为最合理的单点估计。
只输出一个 JSON 对象，恰好包含 fx、fy、cx、cy 四个数值字段，不要解释或 Markdown。"""
FOCAL_PROMPT = """请仅根据这张图片估计拍摄它的相机焦距。
原始图片宽度 W={width} 像素，高度 H={height} 像素。答案必须使用这个原始分辨率的像素单位，即使你的视觉输入被内部缩放。
已知这是针孔透视相机，方形像素、零 skew、无镜头畸变。主点已知位于图像中心，即 cx=W/2、cy=H/2；不需要估计或输出主点。
唯一需要估计的量是 f，满足 fx=fy=f。f 是像素焦距，不是毫米焦距，也不是视场角。
没有提供传感器大小、拍摄距离、物体尺寸、相机位姿或标定板。请结合透视线索，给出你认为最合理的单点估计。
只输出一个 JSON 对象，恰好只含 f 一个字段，值为正数，不要解释、其他字段或 Markdown。"""


def template_for(protocol=PROTOCOL):
    if protocol == PROTOCOL:
        return PROMPT
    if protocol == FOCAL_PROTOCOL:
        return FOCAL_PROMPT
    raise ValueError(f"Unknown intrinsics protocol: {protocol}")


def prompt_for(width, height, protocol=PROTOCOL):
    return template_for(protocol).format(width=width, height=height)


def parse_prediction(text, protocol=PROTOCOL):
    template_for(protocol)  # Reject unknown protocols instead of silently parsing.
    fields = ("f",) if protocol == FOCAL_PROTOCOL else FIELDS
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError("Expected exactly " + ", ".join(fields))
    for field, number in value.items():
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
            raise ValueError(f"Invalid numeric field: {field}")
        if field in {"f", "fx", "fy"} and number <= 0:
            raise ValueError("Focal lengths must be positive")
    return {field: float(value[field]) for field in fields}


def fov_degrees(focal, principal, extent):
    # For an off-axis frustum, 2*atan(extent/(2*focal)) is incorrect.
    return math.degrees(math.atan((extent - principal) / focal) + math.atan(principal / focal))


def baseline(width, height):
    focal = width / (2 * math.tan(math.radians(60) / 2))
    return {"fx": focal, "fy": focal, "cx": width / 2, "cy": height / 2}


def errors(prediction, truth, width, height):
    relative = [abs(prediction[k] - truth[k]) / truth[k] for k in ("fx", "fy")]
    center_px = math.hypot(prediction["cx"] - truth["cx"], prediction["cy"] - truth["cy"])
    center_fraction = center_px / math.hypot(width, height)
    hfov_error = abs(fov_degrees(prediction["fx"], prediction["cx"], width)
                     - fov_degrees(truth["fx"], truth["cx"], width))
    return {"focal_mape_pct": 50 * sum(relative), "focal_max_relative_error": max(relative),
            "principal_error_px": center_px, "principal_error_diagonal_pct": 100 * center_fraction,
            "horizontal_fov_abs_error_deg": hfov_error,
            "focal_within_10pct": max(relative) <= .10,
            "principal_within_2pct_diagonal": center_fraction <= .02,
            "joint_within_10pct_focal_2pct_diagonal": max(relative) <= .10 and center_fraction <= .02}
