"""Shared, Blender-independent definitions for the fixed-camera motion pilot."""
from __future__ import annotations

import json
import math
import re

PROTOCOL = "desktop-fixed-camera-motion-v1"
CAMERA_PROTOCOL = "desktop-changing-camera-motion-v1"
REAL_PROTOCOL = "real-desktop-motion-v1"
DIRECTIONS = ("right", "down_right", "down", "down_left", "left", "up_left", "up", "up_right")
LABELS = dict(zip(DIRECTIONS, ("右", "右下", "下", "左下", "左", "左上", "上", "右上"))) | {"none": "不动"}
TARGETS = {"mug": "带把手的杯子", "bottle": "带盖水瓶", "book": "书", "mouse": "电脑鼠标"}
REAL_TARGETS = {"LeftA": "之前照片中靠左的蓝色字母 A 块（不是玻璃壶旁的另一个 A）",
                "B": "桌面中间的蓝色字母 B 块",
                "C": "桌面右侧的蓝色字母 C 块",
                "jar": "桌面右侧带绿色把手的透明玻璃壶"}
PROMPT_TEMPLATE = """这里有按时间顺序排列的两张图片：图 A 是之前，图 B 是之后。两图的相机位置、朝向、焦距和分辨率相同。
请比较目标物体：{target}。目标可能在桌面上平移，也可能没有移动；不会旋转、缩放或离开桌面。
请仅根据两张图片回答：
1. moved：目标是否发生了位置变化，布尔值。
2. direction：目标从 A 到 B 在画面中的移动方向。以物体整体中心的位置变化为准，右为图像右侧，下为图像下侧；不是物体自身的左右，也不是世界坐标的上下。选择最接近的八方向之一：right（右）、down_right（右下）、down（下）、down_left（左下）、left（左）、up_left（左上）、up（上）、up_right（右上）。没有移动则为 none。
3. distance_cm：目标在桌面上从原位置到新位置的实际直线距离，单位厘米，不是图像上的像素距离。结合场景中日常物体估计，给出最合理的单个数值。
若没有移动，moved 为 false、direction 为 none、distance_cm 为 0；否则 moved 为 true、distance_cm 为正数。
只输出包含 moved、direction、distance_cm 三个字段的 JSON 对象，不要解释或 Markdown。"""


CAMERA_PROMPT_TEMPLATE = """这里有按时间顺序排列的两张图片：图 A 是之前，图 B 是之后。拍摄相机的位置和朝向可能有少量变化，焦距和分辨率不变。
请比较目标物体：{target}。目标可能相对于桌面平移，也可能没有移动；不会旋转、缩放或离开桌面。其他物体相对于桌面保持不动。
请利用桌面和其他物体区分相机变化与目标的真实移动，仅根据两张图片回答：
1. moved：目标相对于桌面是否发生了位置变化，布尔值。仅由相机变化导致的画面位置变化不算目标移动。
2. direction：目标真实移动在图 A 视角中的方向。设想相机仍固定在图 A 的位置和朝向，把目标的新旧两个位置都投影到图 A，以物体整体中心的变化判断；不是直接比较两张原图的像素坐标。右为图 A 的右侧，下为图 A 的下侧。选择最接近的八方向之一：right（右）、down_right（右下）、down（下）、down_left（左下）、left（左）、up_left（左上）、up（上）、up_right（右上）。没有真实移动则为 none。
3. distance_cm：目标相对于桌面从原位置到新位置的实际直线距离，单位厘米，不是像素距离，也不是相机移动距离。结合场景中日常物体估计，给出最合理的单个数值。
若没有移动，moved 为 false、direction 为 none、distance_cm 为 0；否则 moved 为 true、distance_cm 为正数。
只输出包含 moved、direction、distance_cm 三个字段的 JSON 对象，不要解释或 Markdown。"""


REAL_PROMPT_TEMPLATE = """这里有按时间顺序排列的两张真实照片：图 A 是之前，图 B 是之后。相机可能有变化，桌面上的其他物体也可能移动或增减。
请只跟踪指定目标物体：{target}。目标可能相对于桌面移动，也可能没有移动。
请利用桌面和能确认未移动的背景区分相机变化与目标的真实移动，仅根据两张照片回答：
1. moved：指定目标相对于桌面是否发生了位置变化，布尔值。仅由相机变化导致的画面位置变化不算目标移动。
2. direction：目标真实移动在图 A 视角中的方向。设想相机仍固定在图 A 的位置和朝向，把目标的新旧两个位置都投影到图 A，以物体整体中心的变化判断。右为图 A 的右侧，下为图 A 的下侧，不是物体自身的左右。选择最接近的八方向之一：right（右）、down_right（右下）、down（下）、down_left（左下）、left（左）、up_left（左上）、up（上）、up_right（右上）。没有真实移动则为 none。
3. distance_cm：目标相对于桌面从原位置到新位置的实际直线距离，单位厘米，不是像素距离，也不是相机移动距离。结合照片中的场景线索估计，给出最合理的单个数值。
若没有移动，moved 为 false、direction 为 none、distance_cm 为 0；否则 moved 为 true、distance_cm 为正数。
只输出包含 moved、direction、distance_cm 三个字段的 JSON 对象，不要解释或 Markdown。"""


def targets_for(protocol=PROTOCOL):
    if protocol == REAL_PROTOCOL:
        return REAL_TARGETS
    if protocol in {PROTOCOL, CAMERA_PROTOCOL}:
        return TARGETS
    raise ValueError(f"Unknown motion protocol: {protocol}")


def template_for(protocol=PROTOCOL):
    if protocol == PROTOCOL:
        return PROMPT_TEMPLATE
    if protocol == CAMERA_PROTOCOL:
        return CAMERA_PROMPT_TEMPLATE
    if protocol == REAL_PROTOCOL:
        return REAL_PROMPT_TEMPLATE
    raise ValueError(f"Unknown motion protocol: {protocol}")


def prompt_for(target, protocol=PROTOCOL):
    return template_for(protocol).format(target=targets_for(protocol)[target])


def direction_for(dx, dy):
    if math.hypot(dx, dy) < 1e-6:
        return "none"
    return DIRECTIONS[int(math.floor((math.atan2(dy, dx) + math.pi / 8) / (math.pi / 4))) % 8]


def parse_prediction(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {"moved", "direction", "distance_cm"}:
        raise ValueError("Expected exactly moved, direction, distance_cm")
    if type(value["moved"]) is not bool or value["direction"] not in (*DIRECTIONS, "none"):
        raise ValueError("Invalid movement flag or direction")
    number = value["distance_cm"]
    if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or number < 0:
        raise ValueError("Distance must be a finite nonnegative number")
    if value["moved"] != (value["direction"] != "none") or value["moved"] != (number > 0):
        raise ValueError("Movement flag, direction and distance disagree")
    return {**value, "distance_cm": float(number)}
