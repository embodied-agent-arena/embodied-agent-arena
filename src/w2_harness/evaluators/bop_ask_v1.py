"""Deterministic, analysis-only evaluator for BOP-ASK answer contracts.

The evaluator operates on submitted answers and evaluator-side references. It
does not load benchmark assets, call APIs, mutate results, or claim parity with
an official BOP-ASK scorer.
"""

from __future__ import annotations

import ast
import json
import math
import re
import string
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .task_router_v2 import TaskTypeEvaluationResultV2


ANALYSIS_ONLY = True
EVALUATOR_VERSION = "bop_ask_v1.1"

_NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_POINT_CONTAINER_KEYS = (
    "points",
    "object_markers",
    "object markers",
    "bbox",
    "trajectory",
    "waypoints",
    "grasp_points",
)
_GRASP_ROLES = (
    "grasp_center",
    "left_finger_base",
    "right_finger_base",
    "left_finger_tip",
    "right_finger_tip",
)

_RELATION_ALIASES = {
    "left": "left_of",
    "left of": "left_of",
    "to the left of": "left_of",
    "right": "right_of",
    "right of": "right_of",
    "to the right of": "right_of",
    "above": "above",
    "over": "above",
    "below": "below",
    "under": "below",
    "front": "front_of",
    "front of": "front_of",
    "in front of": "front_of",
    "behind": "behind",
    "closer": "closer",
    "closer than": "closer",
    "farther": "farther",
    "farther than": "farther",
    "near": "near",
    "far": "far",
    "overlap": "overlap",
    "overlapping": "overlap",
    "inside": "inside",
    "within": "inside",
    "contains": "contains",
    "disjoint": "disjoint",
    "touching": "touching",
}
_RELATION_INVERSES = {
    "left_of": "right_of",
    "right_of": "left_of",
    "above": "below",
    "below": "above",
    "front_of": "behind",
    "behind": "front_of",
    "closer": "farther",
    "farther": "closer",
    "inside": "contains",
    "contains": "inside",
    "near": "near",
    "far": "far",
    "overlap": "overlap",
    "disjoint": "disjoint",
    "touching": "touching",
}


@dataclass(frozen=True)
class BOPASKMetricConfigV1:
    """Configurable analysis defaults; these are not official thresholds."""

    coordinate_tolerance: float = 5.0
    trajectory_waypoint_tolerance: float = 8.0
    trajectory_endpoint_tolerance: float = 5.0
    bbox_iou_threshold: float = 0.5
    pose_translation_tolerance: float = 0.02
    pose_rotation_tolerance_degrees: float = 5.0
    relation_tolerance: float = 0.0
    relation_distance_threshold: float = 0.1
    coordinate_min_cardinality: int = 1
    coordinate_max_cardinality: int = 100
    grasp_cardinality: int = 5
    bbox_cardinality: int = 8

    def validate(self) -> None:
        non_negative = (
            self.coordinate_tolerance,
            self.trajectory_waypoint_tolerance,
            self.trajectory_endpoint_tolerance,
            self.pose_translation_tolerance,
            self.pose_rotation_tolerance_degrees,
            self.relation_tolerance,
            self.relation_distance_threshold,
        )
        if any(not math.isfinite(value) or value < 0 for value in non_negative):
            raise ValueError("BOP-ASK tolerances must be finite and non-negative")
        if (
            not math.isfinite(self.bbox_iou_threshold)
            or not 0.0 <= self.bbox_iou_threshold <= 1.0
        ):
            raise ValueError("bbox_iou_threshold must be between zero and one")
        if self.coordinate_min_cardinality < 1:
            raise ValueError("coordinate_min_cardinality must be positive")
        if self.coordinate_max_cardinality < self.coordinate_min_cardinality:
            raise ValueError("coordinate cardinality range is invalid")
        if self.grasp_cardinality != len(_GRASP_ROLES):
            raise ValueError("BOP-ASK grasp cardinality must be five")
        if self.bbox_cardinality < 2:
            raise ValueError("bbox_cardinality must be at least two")


@dataclass(frozen=True)
class _ParsedPoints:
    points: tuple[tuple[float, float], ...]
    labels: tuple[str | None, ...]
    source_format: str


@dataclass(frozen=True)
class _Pose:
    translation: tuple[float, float, float]
    rotation: tuple[tuple[float, float, float], ...]
    rotation_representation: str


@dataclass(frozen=True)
class _Relation:
    name: str
    subject: str | None = None
    object: str | None = None


class _EvaluationIssue(ValueError):
    def __init__(
        self,
        status: str,
        reason: str,
        *,
        side: str = "prediction",
        decisive: bool = True,
    ) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason
        self.side = side
        self.decisive = decisive


def _is_blank(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(re.sub(r"[\W_]+", " ", text).split())


def _finite_float(value: object, *, side: str, field: str) -> float:
    if isinstance(value, bool):
        raise _EvaluationIssue(
            "parse_failed",
            f"{field} must be numeric",
            side=side,
        )
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise _EvaluationIssue(
            "parse_failed",
            f"{field} must be numeric",
            side=side,
        ) from exc
    if not math.isfinite(number):
        raise _EvaluationIssue(
            "parse_failed",
            f"{field} must be finite",
            side=side,
        )
    return number


def _metadata_float(
    metadata: Mapping[str, Any],
    key: str,
    default: float,
    *,
    maximum: float | None = None,
) -> float:
    raw = metadata.get(key, default)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{key} must be finite and non-negative")
    if maximum is not None and value > maximum:
        raise ValueError(f"{key} must not exceed {maximum}")
    return value


def _explicit_metadata_float(
    metadata: Mapping[str, Any],
    key: str,
    *,
    maximum: float | None = None,
) -> float | None:
    """Return a validated value only when the metric contract supplies it."""
    if key not in metadata or metadata.get(key) is None:
        return None
    return _metadata_float(metadata, key, 0.0, maximum=maximum)


def _metadata_positive_int(
    metadata: Mapping[str, Any],
    key: str,
    default: int,
) -> int:
    raw = metadata.get(key, default)
    if isinstance(raw, bool):
        raise ValueError(f"{key} must be a positive integer")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a positive integer") from exc
    if value < 1 or value != float(raw):
        raise ValueError(f"{key} must be a positive integer")
    return value


def _decode_structured(value: object, *, side: str) -> tuple[object, str]:
    if not isinstance(value, str):
        return value, "native"
    text = value.strip()
    if not text:
        raise _EvaluationIssue("parse_failed", "empty structured answer", side=side)
    candidates = [text]
    starts = [index for index in (text.find("{"), text.find("[")) if index >= 0]
    if starts:
        suffix = text[min(starts) :].strip()
        if suffix != text:
            candidates.append(suffix)
    for candidate in candidates:
        for loader, source_format in (
            (json.loads, "json"),
            (ast.literal_eval, "python_literal"),
        ):
            try:
                return loader(candidate), source_format
            except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
                continue
    raise _EvaluationIssue(
        "parse_failed",
        "structured answer is neither valid JSON nor a safe literal",
        side=side,
    )


def _point_from_value(
    value: object,
    *,
    side: str,
) -> tuple[tuple[float, float], str | None]:
    label: str | None = None
    if isinstance(value, Mapping):
        if "x" not in value or "y" not in value:
            raise _EvaluationIssue(
                "parse_failed",
                "point mapping requires x and y",
                side=side,
            )
        raw_x, raw_y = value["x"], value["y"]
        raw_label = value.get("label", value.get("name", value.get("role")))
        label = str(raw_label).strip() if raw_label is not None else None
    elif (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 2
    ):
        raw_x, raw_y = value
    else:
        raise _EvaluationIssue(
            "parse_failed",
            "each point must contain exactly x and y",
            side=side,
        )
    return (
        (
            _finite_float(raw_x, side=side, field="x"),
            _finite_float(raw_y, side=side, field="y"),
        ),
        label,
    )


def _points_from_structured(
    value: object,
    *,
    side: str,
    source_format: str,
) -> _ParsedPoints:
    source = value
    mapped_labels: list[str] | None = None
    if isinstance(source, Mapping):
        lowered_keys = {str(key).casefold(): key for key in source}
        container_key = next(
            (
                lowered_keys[key]
                for key in _POINT_CONTAINER_KEYS
                if key in lowered_keys
            ),
            None,
        )
        if container_key is not None:
            source = source[container_key]
        elif "x" in source and "y" in source:
            source = (source,)
        elif source and all(
            isinstance(item, Sequence)
            and not isinstance(item, (str, bytes))
            and len(item) == 2
            for item in source.values()
        ):
            mapped_labels = [str(key) for key in source]
            source = tuple(source.values())
        else:
            raise _EvaluationIssue(
                "parse_failed",
                "point object has no supported coordinate container",
                side=side,
            )
    if not isinstance(source, Sequence) or isinstance(source, (str, bytes)):
        raise _EvaluationIssue(
            "parse_failed",
            "coordinate answer must be a point or point sequence",
            side=side,
        )
    if (
        len(source) == 2
        and not isinstance(source[0], (Sequence, Mapping))
        and not isinstance(source[1], (Sequence, Mapping))
    ):
        source = (source,)
    elif source and all(
        not isinstance(item, (Sequence, Mapping)) for item in source
    ):
        if len(source) % 2:
            raise _EvaluationIssue(
                "parse_failed",
                "flat coordinate arrays require an even number of values",
                side=side,
            )
        source = tuple(zip(source[::2], source[1::2]))
    points: list[tuple[float, float]] = []
    labels: list[str | None] = []
    for index, item in enumerate(source):
        point, label = _point_from_value(item, side=side)
        points.append(point)
        labels.append(mapped_labels[index] if mapped_labels is not None else label)
    if not points:
        raise _EvaluationIssue(
            "invalid_coordinate_contract",
            "coordinate sequence must not be empty",
            side=side,
        )
    return _ParsedPoints(tuple(points), tuple(labels), source_format)


def _parse_xml_points(value: str, *, side: str) -> _ParsedPoints:
    tag_re = re.compile(
        r"<points?\b(?P<attributes>[^>]*)>(?P<label>.*?)</points?>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    matches = tuple(tag_re.finditer(value))
    if not matches:
        raise _EvaluationIssue(
            "parse_failed",
            "point XML has no complete point elements",
            side=side,
        )
    points: list[tuple[float, float]] = []
    labels: list[str | None] = []
    for match in matches:
        attributes = {
            key.casefold(): raw_value
            for key, _, raw_value in re.findall(
                r"([A-Za-z_][\w\-]*)\s*=\s*([\"'])\s*([^\"']+?)\s*\2",
                match.group("attributes"),
            )
        }
        if "x" not in attributes or "y" not in attributes:
            raise _EvaluationIssue(
                "parse_failed",
                "each XML point requires x and y attributes",
                side=side,
            )
        points.append(
            (
                _finite_float(attributes["x"], side=side, field="x"),
                _finite_float(attributes["y"], side=side, field="y"),
            )
        )
        label = re.sub(r"<[^>]+>", "", match.group("label")).strip()
        labels.append(label or None)
    outside = tag_re.sub("", value).strip()
    if outside:
        raise _EvaluationIssue(
            "parse_failed",
            "point XML contains text outside point elements",
            side=side,
        )
    return _ParsedPoints(tuple(points), tuple(labels), "xml")


def _parse_labeled_grasp_points(value: str, *, side: str) -> _ParsedPoints | None:
    role_pattern = "|".join(
        re.escape(role.replace("_", " ")) for role in _GRASP_ROLES
    )
    pattern = re.compile(
        rf"(?P<label>{role_pattern})\s*:\s*"
        rf"\[\s*(?P<x>{_NUMBER_PATTERN})\s*,\s*"
        rf"(?P<y>{_NUMBER_PATTERN})\s*\]",
        flags=re.IGNORECASE,
    )
    matches = tuple(pattern.finditer(value))
    if not matches:
        return None
    outside = pattern.sub("", value)
    if outside.strip(" \t\r\n,;"):
        raise _EvaluationIssue(
            "parse_failed",
            "labeled grasp answer contains unsupported text",
            side=side,
        )
    points: list[tuple[float, float]] = []
    labels: list[str] = []
    for match in matches:
        labels.append(_canonical_label(match.group("label")))
        points.append(
            (
                _finite_float(match.group("x"), side=side, field="x"),
                _finite_float(match.group("y"), side=side, field="y"),
            )
        )
    return _ParsedPoints(tuple(points), tuple(labels), "labeled_grasp_text")


def _parse_points(value: object, *, side: str) -> _ParsedPoints:
    if isinstance(value, str) and re.search(r"<points?\b", value, re.IGNORECASE):
        return _parse_xml_points(value.strip(), side=side)
    if isinstance(value, str):
        parsed_grasp = _parse_labeled_grasp_points(value.strip(), side=side)
        if parsed_grasp is not None:
            return parsed_grasp
    decoded, source_format = _decode_structured(value, side=side)
    return _points_from_structured(
        decoded,
        side=side,
        source_format=source_format,
    )


def _canonical_label(value: str) -> str:
    return "_".join(_normalize_text(value).split())


def _ordered_grasp_points(
    parsed: _ParsedPoints,
    *,
    side: str,
    allow_unlabeled: bool,
) -> tuple[tuple[float, float], ...]:
    labels_present = tuple(label is not None for label in parsed.labels)
    if any(labels_present) and not all(labels_present):
        raise _EvaluationIssue(
            "ambiguous_prediction" if side == "prediction" else "ambiguous_reference",
            "grasp roles are only partially labeled",
            side=side,
            decisive=side == "prediction",
        )
    if not any(labels_present):
        if not allow_unlabeled:
            raise _EvaluationIssue(
                "ambiguous_prediction"
                if side == "prediction"
                else "ambiguous_reference",
                "unlabeled grasp points have no role ordering",
                side=side,
                decisive=side == "prediction",
            )
        return parsed.points
    role_to_point: dict[str, tuple[float, float]] = {}
    for label, point in zip(parsed.labels, parsed.points):
        assert label is not None
        canonical = _canonical_label(label)
        if canonical not in _GRASP_ROLES:
            raise _EvaluationIssue(
                "invalid_coordinate_contract",
                f"unknown grasp role: {label}",
                side=side,
            )
        if canonical in role_to_point:
            raise _EvaluationIssue(
                "ambiguous_prediction"
                if side == "prediction"
                else "ambiguous_reference",
                f"duplicate grasp role: {label}",
                side=side,
                decisive=side == "prediction",
            )
        role_to_point[canonical] = point
    if set(role_to_point) != set(_GRASP_ROLES):
        raise _EvaluationIssue(
            "invalid_coordinate_contract",
            "grasp answer does not contain all required roles",
            side=side,
        )
    return tuple(role_to_point[role] for role in _GRASP_ROLES)


def _ordered_trajectory_points(
    parsed: _ParsedPoints,
    *,
    side: str,
) -> tuple[tuple[float, float], ...]:
    labels_present = tuple(label is not None for label in parsed.labels)
    if any(labels_present) and not all(labels_present):
        raise _EvaluationIssue(
            "ambiguous_prediction" if side == "prediction" else "ambiguous_reference",
            "trajectory labels are only partially specified",
            side=side,
            decisive=side == "prediction",
        )
    if not any(labels_present):
        return parsed.points
    indices: list[int] = []
    for label in parsed.labels:
        assert label is not None
        match = re.fullmatch(r"point[\s_\-]*(\d+)", _normalize_text(label))
        if match is None:
            raise _EvaluationIssue(
                "invalid_sequence",
                f"unsupported trajectory waypoint label: {label}",
                side=side,
            )
        indices.append(int(match.group(1)))
    expected = list(range(1, len(indices) + 1))
    if indices != expected:
        raise _EvaluationIssue(
            "invalid_sequence",
            "trajectory waypoint labels must be contiguous and ordered from point1",
            side=side,
        )
    return parsed.points


def _coordinate_bounds(
    metadata: Mapping[str, Any],
    question: str,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    raw_bounds = metadata.get("coordinate_bounds")
    if isinstance(raw_bounds, Mapping):
        raw_x = raw_bounds.get("x")
        raw_y = raw_bounds.get("y")
        if (
            isinstance(raw_x, Sequence)
            and not isinstance(raw_x, (str, bytes))
            and len(raw_x) == 2
            and isinstance(raw_y, Sequence)
            and not isinstance(raw_y, (str, bytes))
            and len(raw_y) == 2
        ):
            x_bounds = (float(raw_x[0]), float(raw_x[1]))
            y_bounds = (float(raw_y[0]), float(raw_y[1]))
            if (
                all(math.isfinite(value) for value in (*x_bounds, *y_bounds))
                and x_bounds[0] < x_bounds[1]
                and y_bounds[0] < y_bounds[1]
            ):
                return x_bounds, y_bounds
            raise ValueError("coordinate_bounds must contain finite increasing pairs")
        raise ValueError("coordinate_bounds requires x and y pairs")
    width = metadata.get("image_width")
    height = metadata.get("image_height")
    if width is not None or height is not None:
        if width is None or height is None:
            raise ValueError("image_width and image_height must be provided together")
        parsed_width = float(width)
        parsed_height = float(height)
        if (
            not math.isfinite(parsed_width)
            or not math.isfinite(parsed_height)
            or parsed_width <= 0
            or parsed_height <= 0
        ):
            raise ValueError("image dimensions must be finite and positive")
        return (0.0, parsed_width), (0.0, parsed_height)
    normalized = " ".join(str(question or "").split())
    x_match = re.search(
        rf"\bx\b[^\n]{{0,80}}?between\s*\(\s*({_NUMBER_PATTERN})\s*,\s*"
        rf"({_NUMBER_PATTERN})\s*\)",
        normalized,
        flags=re.IGNORECASE,
    )
    y_match = re.search(
        rf"\by\b[^\n]{{0,80}}?between\s*\(\s*({_NUMBER_PATTERN})\s*,\s*"
        rf"({_NUMBER_PATTERN})\s*\)",
        normalized,
        flags=re.IGNORECASE,
    )
    if x_match and y_match:
        return (
            (float(x_match.group(1)), float(x_match.group(2))),
            (float(y_match.group(1)), float(y_match.group(2))),
        )
    return None


def _validate_points_contract(
    points: tuple[tuple[float, float], ...],
    *,
    metadata: Mapping[str, Any],
    question: str,
    side: str,
    exact_cardinality: int | None = None,
) -> dict[str, Any]:
    minimum = _metadata_positive_int(
        metadata,
        "coordinate_min_cardinality",
        1,
    )
    maximum = _metadata_positive_int(
        metadata,
        "coordinate_max_cardinality",
        100,
    )
    if maximum < minimum:
        raise ValueError("coordinate cardinality metadata is invalid")
    exact = metadata.get(
        "coordinate_cardinality",
        metadata.get("expected_cardinality", exact_cardinality),
    )
    if exact is not None:
        exact = _metadata_positive_int(
            {"value": exact},
            "value",
            exact_cardinality or 1,
        )
    count_valid = minimum <= len(points) <= maximum
    if exact is not None:
        count_valid = count_valid and len(points) == exact
    bounds = _coordinate_bounds(metadata, question)
    range_valid = True
    details: dict[str, Any] = {
        "cardinality": len(points),
        "cardinality_valid": count_valid,
    }
    if bounds is not None:
        x_bounds, y_bounds = bounds
        range_valid = all(
            x_bounds[0] <= x < x_bounds[1]
            and y_bounds[0] <= y < y_bounds[1]
            for x, y in points
        )
        details.update(
            {
                "x_range": x_bounds,
                "y_range": y_bounds,
                "range_valid": range_valid,
                "bounds_semantics": "lower_inclusive_upper_exclusive",
            }
        )
    if not count_valid:
        raise _EvaluationIssue(
            "cardinality_mismatch",
            "coordinate cardinality does not match the BOP-ASK contract",
            side=side,
        )
    if not range_valid:
        raise _EvaluationIssue(
            "invalid_coordinate_contract",
            "coordinate lies outside the declared image bounds",
            side=side,
        )
    return details


def _euclidean_errors(
    prediction: Sequence[tuple[float, float]],
    ground_truth: Sequence[tuple[float, float]],
) -> tuple[float, ...]:
    return tuple(
        math.hypot(pred_x - truth_x, pred_y - truth_y)
        for (pred_x, pred_y), (truth_x, truth_y) in zip(
            prediction,
            ground_truth,
        )
    )


def _has_perfect_matching(
    distances: Sequence[Sequence[float]],
    threshold: float,
) -> bool:
    matched_prediction = [-1] * len(distances)

    def assign(reference_index: int, seen: set[int]) -> bool:
        for prediction_index, distance in enumerate(distances[reference_index]):
            if distance > threshold or prediction_index in seen:
                continue
            seen.add(prediction_index)
            previous = matched_prediction[prediction_index]
            if previous < 0 or assign(previous, seen):
                matched_prediction[prediction_index] = reference_index
                return True
        return False

    return all(assign(index, set()) for index in range(len(distances)))


def _minimum_bottleneck_error(
    prediction: Sequence[tuple[float, float]],
    ground_truth: Sequence[tuple[float, float]],
) -> float:
    distances = tuple(
        tuple(
            math.hypot(pred_x - truth_x, pred_y - truth_y)
            for pred_x, pred_y in prediction
        )
        for truth_x, truth_y in ground_truth
    )
    candidates = sorted({distance for row in distances for distance in row})
    low, high = 0, len(candidates) - 1
    while low < high:
        middle = (low + high) // 2
        if _has_perfect_matching(distances, candidates[middle]):
            high = middle
        else:
            low = middle + 1
    return candidates[low] if candidates else 0.0


def _bbox_from_points(
    points: Sequence[tuple[float, float]],
    *,
    side: str,
) -> tuple[float, float, float, float]:
    xs = tuple(point[0] for point in points)
    ys = tuple(point[1] for point in points)
    box = (min(xs), min(ys), max(xs), max(ys))
    if box[0] >= box[2] or box[1] >= box[3]:
        raise _EvaluationIssue(
            "invalid_bbox",
            "bbox envelope must have positive area",
            side=side,
        )
    return box


def _bbox_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(
        0.0,
        min(first[3], second[3]) - max(first[1], second[1]),
    )
    intersection = intersection_width * intersection_height
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _point_on_segment(
    point: tuple[float, float],
    first: tuple[float, float],
    second: tuple[float, float],
) -> bool:
    cross = (
        (point[1] - first[1]) * (second[0] - first[0])
        - (point[0] - first[0]) * (second[1] - first[1])
    )
    if abs(cross) > 1e-9:
        return False
    return (
        min(first[0], second[0]) - 1e-9
        <= point[0]
        <= max(first[0], second[0]) + 1e-9
        and min(first[1], second[1]) - 1e-9
        <= point[1]
        <= max(first[1], second[1]) + 1e-9
    )


def _point_in_polygon(
    point: tuple[float, float],
    vertices: Sequence[tuple[float, float]],
) -> bool:
    if len(vertices) < 3:
        raise ValueError("polygon region must contain at least three vertices")
    inside = False
    previous = vertices[-1]
    for current in vertices:
        if _point_on_segment(point, previous, current):
            return True
        if (current[1] > point[1]) != (previous[1] > point[1]):
            x_crossing = (
                (previous[0] - current[0])
                * (point[1] - current[1])
                / (previous[1] - current[1])
                + current[0]
            )
            if point[0] < x_crossing:
                inside = not inside
        previous = current
    return inside


def _point_in_region(
    point: tuple[float, float],
    region: Mapping[str, Any],
) -> bool:
    """Evaluate a point against a sanitized evaluator-side region contract."""
    region_type = str(region.get("type") or "").casefold()
    if region_type == "bbox":
        bounds = region.get("bounds")
        if (
            not isinstance(bounds, Sequence)
            or isinstance(bounds, (str, bytes))
            or len(bounds) != 4
        ):
            raise ValueError("bbox region requires [xmin, ymin, xmax, ymax]")
        xmin, ymin, xmax, ymax = (
            _finite_float(value, side="reference", field="region bounds")
            for value in bounds
        )
        if xmin >= xmax or ymin >= ymax:
            raise ValueError("bbox region bounds must be increasing")
        return xmin <= point[0] < xmax and ymin <= point[1] < ymax
    if region_type == "polygon":
        vertices = _parse_points(
            region.get("points"),
            side="reference",
        ).points
        return _point_in_polygon(point, vertices)
    if region_type == "mask_rle":
        width = _metadata_positive_int(region, "width", 1)
        height = _metadata_positive_int(region, "height", 1)
        x = int(round(point[0]))
        y = int(round(point[1]))
        if not (0 <= x < width and 0 <= y < height):
            return False
        runs = region.get("runs")
        if not isinstance(runs, Sequence) or isinstance(runs, (str, bytes)):
            raise ValueError("mask_rle region requires row runs")
        for run in runs:
            if (
                not isinstance(run, Sequence)
                or isinstance(run, (str, bytes))
                or len(run) != 3
            ):
                raise ValueError("mask_rle run must be [y, x_start, x_end)")
            run_y, start, end = (int(value) for value in run)
            if run_y == y and start <= x < end:
                return True
        return False
    raise ValueError("region type must be bbox, polygon, or mask_rle")


def _region_matches(
    point: tuple[float, float],
    regions: Sequence[Mapping[str, Any]],
) -> tuple[int, ...]:
    return tuple(
        index
        for index, region in enumerate(regions)
        if _point_in_region(point, region)
    )


def _decode_pose_mapping(value: object, *, side: str) -> Mapping[str, Any]:
    decoded, _ = _decode_structured(value, side=side)
    if not isinstance(decoded, Mapping):
        raise _EvaluationIssue(
            "parse_failed",
            "pose answer must be an object",
            side=side,
        )
    return decoded


def _vector(
    value: object,
    length: int,
    *,
    side: str,
    field: str,
) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise _EvaluationIssue(
            "parse_failed",
            f"{field} must be a length-{length} sequence",
            side=side,
        )
    if len(value) != length:
        raise _EvaluationIssue(
            "parse_failed",
            f"{field} must contain {length} values",
            side=side,
        )
    return tuple(
        _finite_float(item, side=side, field=field)
        for item in value
    )


def _quaternion_to_matrix(
    values: Sequence[float],
    *,
    order: str,
    side: str,
) -> tuple[tuple[float, float, float], ...]:
    if order == "xyzw":
        x, y, z, w = values
    elif order == "wxyz":
        w, x, y, z = values
    else:
        raise ValueError("quaternion_order must be xyzw or wxyz")
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        raise _EvaluationIssue(
            "parse_failed",
            "rotation quaternion has zero norm",
            side=side,
        )
    x, y, z, w = (item / norm for item in (x, y, z, w))
    return (
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
        ),
        (
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
        ),
        (
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
    )


def _matrix_multiply(
    first: Sequence[Sequence[float]],
    second: Sequence[Sequence[float]],
) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        tuple(
            sum(first[row][index] * second[index][column] for index in range(3))
            for column in range(3)
        )
        for row in range(3)
    )


def _euler_xyz_to_matrix(
    values: Sequence[float],
) -> tuple[tuple[float, float, float], ...]:
    x, y, z = values
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)
    rotation_x = ((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx))
    rotation_y = ((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy))
    rotation_z = ((cz, -sz, 0.0), (sz, cz, 0.0), (0.0, 0.0, 1.0))
    return _matrix_multiply(rotation_z, _matrix_multiply(rotation_y, rotation_x))


def _matrix_determinant(matrix: Sequence[Sequence[float]]) -> float:
    return (
        matrix[0][0]
        * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1]
        * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2]
        * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


def _validated_rotation_matrix(
    matrix: Sequence[Sequence[float]],
    *,
    side: str,
) -> tuple[tuple[float, float, float], ...]:
    rows = tuple(tuple(row) for row in matrix)
    row_norms = tuple(math.sqrt(sum(value * value for value in row)) for row in rows)
    row_dots = tuple(
        sum(rows[first][index] * rows[second][index] for index in range(3))
        for first, second in ((0, 1), (0, 2), (1, 2))
    )
    determinant = _matrix_determinant(rows)
    if (
        any(abs(norm - 1.0) > 1e-3 for norm in row_norms)
        or any(abs(dot) > 1e-3 for dot in row_dots)
        or abs(determinant - 1.0) > 1e-3
    ):
        raise _EvaluationIssue(
            "parse_failed",
            "rotation matrix is not a valid SO(3) matrix",
            side=side,
        )
    return rows


def _parse_rotation(
    value: object,
    *,
    metadata: Mapping[str, Any],
    side: str,
) -> tuple[tuple[tuple[float, float, float], ...], str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise _EvaluationIssue(
            "parse_failed",
            "rotation must be a sequence",
            side=side,
        )
    if (
        len(value) == 3
        and all(
            isinstance(row, Sequence)
            and not isinstance(row, (str, bytes))
            and len(row) == 3
            for row in value
        )
    ):
        matrix = tuple(
            tuple(
                _finite_float(item, side=side, field="rotation")
                for item in row
            )
            for row in value
        )
        return _validated_rotation_matrix(matrix, side=side), "matrix"
    flat = tuple(value)
    if len(flat) == 9:
        numbers = tuple(
            _finite_float(item, side=side, field="rotation")
            for item in flat
        )
        matrix = tuple(
            numbers[index : index + 3]
            for index in range(0, 9, 3)
        )
        return _validated_rotation_matrix(matrix, side=side), "matrix"
    if len(flat) == 4:
        quaternion = tuple(
            _finite_float(item, side=side, field="rotation")
            for item in flat
        )
        order = str(metadata.get("quaternion_order", "xyzw")).casefold()
        return (
            _quaternion_to_matrix(
                quaternion,
                order=order,
                side=side,
            ),
            f"quaternion_{order}",
        )
    if len(flat) == 3:
        representation = str(
            metadata.get("rotation_representation") or ""
        ).casefold()
        if representation not in {
            "euler_xyz_degrees",
            "euler_xyz_radians",
        }:
            raise _EvaluationIssue(
                "ambiguous_prediction"
                if side == "prediction"
                else "ambiguous_reference",
                "three-value rotation requires an explicit Euler representation",
                side=side,
                decisive=side == "prediction",
            )
        angles = tuple(
            _finite_float(item, side=side, field="rotation")
            for item in flat
        )
        if representation.endswith("degrees"):
            angles = tuple(math.radians(item) for item in angles)
        return _euler_xyz_to_matrix(angles), representation
    raise _EvaluationIssue(
        "parse_failed",
        "rotation must be a quaternion, 3x3 matrix, or explicit Euler vector",
        side=side,
    )


def _parse_pose(
    value: object,
    *,
    metadata: Mapping[str, Any],
    side: str,
) -> _Pose:
    mapping = _decode_pose_mapping(value, side=side)
    translation_value = next(
        (
            mapping[key]
            for key in ("translation", "position", "t")
            if key in mapping
        ),
        None,
    )
    rotation_value = next(
        (
            mapping[key]
            for key in ("rotation", "orientation", "quaternion", "R")
            if key in mapping
        ),
        None,
    )
    if translation_value is None or rotation_value is None:
        raise _EvaluationIssue(
            "parse_failed",
            "pose requires translation and rotation",
            side=side,
        )
    translation = _vector(
        translation_value,
        3,
        side=side,
        field="translation",
    )
    rotation, representation = _parse_rotation(
        rotation_value,
        metadata=metadata,
        side=side,
    )
    return _Pose(translation, rotation, representation)


def _rotation_error_degrees(
    prediction: Sequence[Sequence[float]],
    ground_truth: Sequence[Sequence[float]],
) -> float:
    relative = tuple(
        tuple(
            sum(
                prediction[row][index] * ground_truth[column][index]
                for index in range(3)
            )
            for column in range(3)
        )
        for row in range(3)
    )
    trace = relative[0][0] + relative[1][1] + relative[2][2]
    cosine = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    return math.degrees(math.acos(cosine))


def _choice_map(
    choices: Iterable[object] | Mapping[object, object],
) -> dict[str, str]:
    if isinstance(choices, Mapping):
        return {
            str(label).strip().upper(): str(text).strip()
            for label, text in choices.items()
        }
    if isinstance(choices, str):
        source: Sequence[object] = (choices,)
    else:
        source = tuple(choices)
    mapping: dict[str, str] = {}
    for index, raw in enumerate(source):
        fallback = string.ascii_uppercase[index] if index < 26 else str(index + 1)
        if isinstance(raw, Mapping):
            label = str(raw.get("label", fallback)).strip().upper()
            text = str(raw.get("text", raw.get("value", ""))).strip()
        else:
            text = str(raw).strip()
            match = re.match(
                r"^\s*(?:option\s+)?[\(\[]?([A-Za-z]|\d+)[\)\]]?"
                r"(?:\s*[\.\:\)\-]\s*|\s+$)",
                text,
                flags=re.IGNORECASE,
            )
            if match and text[match.end() :].strip():
                label = match.group(1).upper()
                text = text[match.end() :].strip()
            else:
                label = fallback
        mapping[label] = text
    return mapping


def _yes_no(
    value: object,
    choices: Iterable[object] | Mapping[object, object],
) -> bool | None:
    text = str(value or "").strip()
    mapping = _choice_map(choices)
    label_match = re.fullmatch(
        r"(?:the\s+)?(?:answer|option|choice)?\s*(?:is|:)?\s*"
        r"[\(\[]?([A-Za-z]|\d+)[\)\]]?[\.\s]*",
        text,
        flags=re.IGNORECASE,
    )
    if label_match:
        resolved = mapping.get(label_match.group(1).upper())
        if resolved is not None:
            text = resolved
    normalized = _normalize_text(text)
    first = normalized.split(" ", 1)[0] if normalized else ""
    if first in {"yes", "y", "true", "affirmative"}:
        return True
    if first in {"no", "n", "false", "negative"}:
        return False
    return None


def _parse_relation(value: object, *, side: str) -> _Relation | None:
    subject: str | None = None
    object_name: str | None = None
    raw_relation: object = value
    if isinstance(value, Mapping):
        raw_relation = value.get("relation", value.get("predicate"))
        subject_value = value.get("subject")
        object_value = value.get("object")
        subject = _normalize_text(subject_value) or None
        object_name = _normalize_text(object_value) or None
    if raw_relation is None:
        return None
    normalized = _normalize_text(raw_relation)
    direct = _RELATION_ALIASES.get(normalized)
    if direct is not None:
        return _Relation(direct, subject, object_name)
    matches: set[str] = set()
    for alias in sorted(_RELATION_ALIASES, key=len, reverse=True):
        alias_normalized = _normalize_text(alias)
        if re.search(rf"\b{re.escape(alias_normalized)}\b", normalized):
            matches.add(_RELATION_ALIASES[alias])
    if len(matches) > 1:
        raise _EvaluationIssue(
            "ambiguous_prediction"
            if side == "prediction"
            else "ambiguous_reference",
            "answer contains multiple incompatible spatial relations",
            side=side,
            decisive=side == "prediction",
        )
    if not matches:
        return None
    return _Relation(next(iter(matches)), subject, object_name)


def _relations_match(prediction: _Relation, ground_truth: _Relation) -> bool:
    if (
        prediction.subject
        and prediction.object
        and ground_truth.subject
        and ground_truth.object
    ):
        if (
            prediction.subject == ground_truth.subject
            and prediction.object == ground_truth.object
        ):
            return prediction.name == ground_truth.name
        if (
            prediction.subject == ground_truth.object
            and prediction.object == ground_truth.subject
        ):
            return _RELATION_INVERSES.get(prediction.name) == ground_truth.name
        return False
    return prediction.name == ground_truth.name


def _geometry_point(value: object, *, field: str) -> tuple[float, ...]:
    if isinstance(value, Mapping):
        ordered = tuple(
            value[key]
            for key in ("x", "y", "z")
            if key in value
        )
        value = ordered
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a coordinate sequence")
    if len(value) not in {1, 2, 3}:
        raise ValueError(f"{field} must have one, two, or three coordinates")
    coordinates = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in coordinates):
        raise ValueError(f"{field} coordinates must be finite")
    return coordinates


def _geometry_bbox(value: object, *, field: str) -> tuple[float, float, float, float]:
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 4
        and all(not isinstance(item, Sequence) for item in value)
    ):
        box = tuple(float(item) for item in value)
    else:
        parsed = _points_from_structured(
            value,
            side="reference",
            source_format="relation_geometry",
        )
        box = _bbox_from_points(parsed.points, side="reference")
    if (
        len(box) != 4
        or not all(math.isfinite(item) for item in box)
        or box[0] >= box[2]
        or box[1] >= box[3]
    ):
        raise ValueError(f"{field} must be a positive-area xyxy bbox")
    return box  # type: ignore[return-value]


def _evaluate_relation_geometry(
    geometry: Mapping[str, Any],
    *,
    config: BOPASKMetricConfigV1,
    metadata: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    relation = _parse_relation(
        geometry.get("relation", metadata.get("relation")),
        side="reference",
    )
    if relation is None:
        raise ValueError("relation_geometry requires a supported relation")
    subject = geometry.get(
        "subject",
        geometry.get("subject_geometry", geometry.get("subject_position")),
    )
    object_value = geometry.get(
        "object",
        geometry.get("object_geometry", geometry.get("object_position")),
    )
    if subject is None or object_value is None:
        raise ValueError("relation_geometry requires subject and object geometry")
    tolerance = _metadata_float(
        {**metadata, **geometry},
        "relation_tolerance",
        config.relation_tolerance,
    )
    threshold = _metadata_float(
        {**metadata, **geometry},
        "relation_distance_threshold",
        config.relation_distance_threshold,
    )
    relation_name = relation.name
    details: dict[str, Any] = {
        "relation": relation_name,
        "relation_tolerance": tolerance,
    }
    if relation_name in {
        "left_of",
        "right_of",
        "above",
        "below",
        "front_of",
        "behind",
        "closer",
        "farther",
        "near",
        "far",
    }:
        subject_point = _geometry_point(subject, field="subject")
        object_point = _geometry_point(object_value, field="object")
        if len(subject_point) != len(object_point):
            raise ValueError("relation geometry coordinate dimensions must match")
        if relation_name == "left_of":
            result = subject_point[0] < object_point[0] - tolerance
            margin = object_point[0] - subject_point[0]
        elif relation_name == "right_of":
            result = subject_point[0] > object_point[0] + tolerance
            margin = subject_point[0] - object_point[0]
        elif relation_name == "above":
            if len(subject_point) < 2:
                raise ValueError("above relation requires 2D or 3D points")
            result = subject_point[1] < object_point[1] - tolerance
            margin = object_point[1] - subject_point[1]
        elif relation_name == "below":
            if len(subject_point) < 2:
                raise ValueError("below relation requires 2D or 3D points")
            result = subject_point[1] > object_point[1] + tolerance
            margin = subject_point[1] - object_point[1]
        elif relation_name in {"front_of", "closer"}:
            result = subject_point[-1] < object_point[-1] - tolerance
            margin = object_point[-1] - subject_point[-1]
        elif relation_name in {"behind", "farther"}:
            result = subject_point[-1] > object_point[-1] + tolerance
            margin = subject_point[-1] - object_point[-1]
        else:
            distance = math.sqrt(
                sum(
                    (subject_item - object_item) ** 2
                    for subject_item, object_item in zip(
                        subject_point,
                        object_point,
                    )
                )
            )
            result = (
                distance <= threshold + tolerance
                if relation_name == "near"
                else distance > threshold + tolerance
            )
            margin = distance
            details["relation_distance_threshold"] = threshold
        details["geometry_margin"] = margin
        return result, relation_name, details
    subject_bbox = _geometry_bbox(subject, field="subject")
    object_bbox = _geometry_bbox(object_value, field="object")
    iou = _bbox_iou(subject_bbox, object_bbox)
    details["bbox_iou"] = iou
    if relation_name == "overlap":
        result = iou > threshold
        details["relation_distance_threshold"] = threshold
    elif relation_name == "disjoint":
        result = iou == 0.0
    elif relation_name == "inside":
        result = (
            subject_bbox[0] >= object_bbox[0] - tolerance
            and subject_bbox[1] >= object_bbox[1] - tolerance
            and subject_bbox[2] <= object_bbox[2] + tolerance
            and subject_bbox[3] <= object_bbox[3] + tolerance
        )
    elif relation_name == "contains":
        result = (
            object_bbox[0] >= subject_bbox[0] - tolerance
            and object_bbox[1] >= subject_bbox[1] - tolerance
            and object_bbox[2] <= subject_bbox[2] + tolerance
            and object_bbox[3] <= subject_bbox[3] + tolerance
        )
    elif relation_name == "touching":
        horizontal_gap = max(
            0.0,
            max(subject_bbox[0], object_bbox[0])
            - min(subject_bbox[2], object_bbox[2]),
        )
        vertical_gap = max(
            0.0,
            max(subject_bbox[1], object_bbox[1])
            - min(subject_bbox[3], object_bbox[3]),
        )
        result = math.hypot(horizontal_gap, vertical_gap) <= tolerance
    else:
        raise ValueError(f"unsupported relation geometry: {relation_name}")
    return result, relation_name, details


class BOPASKEvaluatorV1:
    """Evaluate BOP-ASK selection formats at an offline analysis boundary."""

    def __init__(self, config: BOPASKMetricConfigV1 | None = None) -> None:
        self.config = config or BOPASKMetricConfigV1()
        self.config.validate()

    def infer_family(self, metadata: Mapping[str, Any]) -> str:
        explicit = str(
            metadata.get("bop_metric_family")
            or metadata.get("bop_ask_family")
            or ""
        ).casefold()
        aliases = {
            "coordinates": "coordinate",
            "coordinate_set": "coordinate_set",
            "rearrangement": "coordinate_set",
            "2dplane": "grasp",
            "2dbbox": "bbox",
            "6dpose": "pose",
            "relative_position": "relation",
        }
        explicit = aliases.get(explicit, explicit)
        if explicit in {
            "grasp",
            "coordinate",
            "coordinate_set",
            "trajectory",
            "bbox",
            "pose",
            "relation",
        }:
            return explicit
        question_type = str(metadata.get("question_type") or "").casefold()
        subtype = str(metadata.get("question_subtype") or "").casefold()
        if question_type == "grasp" or subtype == "2dplane":
            return "grasp"
        if question_type == "trajectory" or subtype == "2d":
            return "trajectory"
        if question_type == "object_rearrangement" or subtype == "point_wise":
            return "coordinate_set"
        if question_type == "pose" and subtype == "2dbbox":
            return "bbox"
        if question_type == "pose":
            return "pose"
        if question_type in {"depth_relative", "spatial_reasoning"}:
            return "relation"
        return "unsupported"

    def evaluate(
        self,
        prediction: object,
        ground_truth: object | None,
        *,
        answer_type: str = "short_text",
        evaluator: str = "bop_ask_v1",
        choices: Iterable[object] | Mapping[object, object] = (),
        question: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> TaskTypeEvaluationResultV2:
        del answer_type, evaluator
        routed_metadata = dict(metadata or {})
        family = self.infer_family(routed_metadata)
        task_type = (
            "coordinate"
            if family in {
                "grasp",
                "coordinate",
                "coordinate_set",
                "trajectory",
                "bbox",
            }
            else "geometry"
        )
        metric = {
            "grasp": "bop_grasp_ordered_tolerance",
            "coordinate": "bop_coordinate_ordered_tolerance",
            "coordinate_set": "bop_coordinate_set_tolerance",
            "trajectory": "bop_trajectory_sequence_tolerance",
            "bbox": "bop_bbox_iou",
            "pose": "bop_pose_translation_rotation_tolerance",
            "relation": "bop_relation_geometry",
            "unsupported": "bop_ask_unsupported",
        }[family]
        if _is_blank(prediction):
            return self._result(
                family=family,
                task_type=task_type,
                metric=metric,
                score=0.0,
                passed=False,
                status="invalid_prediction",
                prediction_normalized="",
                ground_truth_normalized=None,
                details={"reason": "empty_prediction"},
            )
        relation_geometry = routed_metadata.get("relation_geometry")
        if _is_blank(ground_truth) and not (
            family == "relation" and isinstance(relation_geometry, Mapping)
        ):
            return self._result(
                family=family,
                task_type=task_type,
                metric=metric,
                score=None,
                passed=None,
                status="reference_unavailable",
                prediction_normalized=str(prediction).strip(),
                ground_truth_normalized=None,
            )
        if family == "unsupported":
            return self._result(
                family=family,
                task_type=task_type,
                metric=metric,
                score=None,
                passed=None,
                status="unsupported_bop_task",
                prediction_normalized=str(prediction).strip(),
                ground_truth_normalized=None,
            )
        try:
            method = getattr(self, f"_evaluate_{family}")
            return method(
                prediction,
                ground_truth,
                choices=choices,
                question=question,
                metadata=routed_metadata,
            )
        except _EvaluationIssue as issue:
            passed = (
                False
                if issue.side == "prediction" and issue.decisive
                else None
            )
            return self._result(
                family=family,
                task_type=task_type,
                metric=metric,
                score=0.0 if passed is False else None,
                passed=passed,
                status=issue.status,
                prediction_normalized=None,
                ground_truth_normalized=None,
                details={
                    "reason": issue.reason,
                    "failure_side": issue.side,
                },
            )

    def _result(
        self,
        *,
        family: str,
        task_type: str,
        metric: str,
        score: float | None,
        passed: bool | None,
        status: str,
        prediction_normalized: Any,
        ground_truth_normalized: Any,
        details: Mapping[str, Any] | None = None,
    ) -> TaskTypeEvaluationResultV2:
        return TaskTypeEvaluationResultV2(
            task_type=task_type,
            metric=metric,
            score=score,
            passed=passed,
            status=status,
            prediction_normalized=prediction_normalized,
            ground_truth_normalized=ground_truth_normalized,
            details={
                "analysis_only": True,
                "evaluator_version": EVALUATOR_VERSION,
                "bop_family": family,
                **dict(details or {}),
            },
        )

    def _evaluate_grasp(
        self,
        prediction: object,
        ground_truth: object,
        *,
        choices: Iterable[object] | Mapping[object, object],
        question: str,
        metadata: Mapping[str, Any],
    ) -> TaskTypeEvaluationResultV2:
        del choices
        truth_parsed = _parse_points(ground_truth, side="reference")
        pred_parsed = _parse_points(prediction, side="prediction")
        cardinality = _metadata_positive_int(
            metadata,
            "grasp_cardinality",
            self.config.grasp_cardinality,
        )
        truth = _ordered_grasp_points(
            truth_parsed,
            side="reference",
            allow_unlabeled=True,
        )
        pred = _ordered_grasp_points(
            pred_parsed,
            side="prediction",
            allow_unlabeled=bool(metadata.get("allow_unlabeled_grasp_points", False)),
        )
        _validate_points_contract(
            truth,
            metadata=metadata,
            question=question,
            side="reference",
            exact_cardinality=cardinality,
        )
        details = _validate_points_contract(
            pred,
            metadata=metadata,
            question=question,
            side="prediction",
            exact_cardinality=cardinality,
        )
        errors = _euclidean_errors(pred, truth)
        maximum_error = max(errors, default=0.0)
        mean_error = sum(errors) / len(errors)
        tolerance = _explicit_metadata_float(metadata, "grasp_tolerance")
        if tolerance is None:
            tolerance = _explicit_metadata_float(metadata, "coordinate_tolerance")
        gripper_width = _explicit_metadata_float(
            metadata,
            "gripper_width_pixels",
        )
        nce = (
            mean_error / gripper_width
            if gripper_width is not None and gripper_width > 0
            else None
        )
        nce_threshold = _explicit_metadata_float(
            metadata,
            "grasp_nce_threshold",
        )
        if tolerance is None and (nce is None or nce_threshold is None):
            return self._result(
                family="grasp",
                task_type="coordinate",
                metric="bop_grasp_normalized_coordinate_error",
                score=nce,
                passed=None,
                status="metric_computed_threshold_unpinned",
                prediction_normalized=pred,
                ground_truth_normalized=truth,
                details={
                    **details,
                    "coordinate_order": _GRASP_ROLES,
                    "per_role_errors_pixels": dict(zip(_GRASP_ROLES, errors)),
                    "mean_role_error_pixels": mean_error,
                    "maximum_matching_error_pixels": maximum_error,
                    "normalized_coordinate_error": nce,
                    "threshold_source": "unavailable",
                    "uncertainty": (
                        "official pass threshold and/or gripper-width normalizer "
                        "is not pinned"
                    ),
                },
            )
        if tolerance is not None:
            passed = maximum_error <= tolerance
            threshold_source = (
                "grasp_tolerance"
                if "grasp_tolerance" in metadata
                else "coordinate_tolerance"
            )
        else:
            passed = bool(nce is not None and nce <= nce_threshold)
            threshold_source = "grasp_nce_threshold"
        return self._result(
            family="grasp",
            task_type="coordinate",
            metric="bop_grasp_ordered_tolerance",
            score=float(passed),
            passed=passed,
            status="evaluated",
            prediction_normalized=pred,
            ground_truth_normalized=truth,
            details={
                **details,
                "coordinate_order": _GRASP_ROLES,
                "coordinate_tolerance": tolerance,
                "per_role_errors_pixels": dict(zip(_GRASP_ROLES, errors)),
                "mean_role_error_pixels": mean_error,
                "maximum_matching_error_pixels": maximum_error,
                "normalized_coordinate_error": nce,
                "gripper_width_pixels": gripper_width,
                "grasp_nce_threshold": nce_threshold,
                "threshold_source": threshold_source,
            },
        )

    def _evaluate_coordinate(
        self,
        prediction: object,
        ground_truth: object,
        *,
        choices: Iterable[object] | Mapping[object, object],
        question: str,
        metadata: Mapping[str, Any],
    ) -> TaskTypeEvaluationResultV2:
        return self._evaluate_coordinate_common(
            prediction,
            ground_truth,
            choices=choices,
            question=question,
            metadata=metadata,
            match_mode="ordered",
            family="coordinate",
        )

    def _evaluate_coordinate_set(
        self,
        prediction: object,
        ground_truth: object,
        *,
        choices: Iterable[object] | Mapping[object, object],
        question: str,
        metadata: Mapping[str, Any],
    ) -> TaskTypeEvaluationResultV2:
        object_regions = metadata.get("object_regions")
        if isinstance(object_regions, Sequence) and not isinstance(
            object_regions,
            (str, bytes),
        ):
            return self._evaluate_coordinate_regions(
                prediction,
                ground_truth,
                question=question,
                metadata=metadata,
                object_regions=object_regions,
            )
        return self._evaluate_coordinate_common(
            prediction,
            ground_truth,
            choices=choices,
            question=question,
            metadata=metadata,
            match_mode="set",
            family="coordinate_set",
        )

    def _evaluate_coordinate_regions(
        self,
        prediction: object,
        ground_truth: object,
        *,
        question: str,
        metadata: Mapping[str, Any],
        object_regions: Sequence[Any],
    ) -> TaskTypeEvaluationResultV2:
        if not object_regions or not all(
            isinstance(region, Mapping) for region in object_regions
        ):
            raise ValueError("object_regions must contain region mappings")
        regions = tuple(object_regions)
        truth = _parse_points(ground_truth, side="reference").points
        pred = _parse_points(prediction, side="prediction").points
        _validate_points_contract(
            truth,
            metadata=metadata,
            question=question,
            side="reference",
        )
        details = _validate_points_contract(
            pred,
            metadata=metadata,
            question=question,
            side="prediction",
        )
        truth_matches = tuple(_region_matches(point, regions) for point in truth)
        if (
            any(len(matches) != 1 for matches in truth_matches)
            or len({matches[0] for matches in truth_matches}) != len(regions)
        ):
            raise _EvaluationIssue(
                "ambiguous_reference",
                "reference markers do not uniquely cover every required object region",
                side="reference",
                decisive=False,
            )
        prediction_matches = tuple(_region_matches(point, regions) for point in pred)
        if any(len(matches) > 1 for matches in prediction_matches):
            raise _EvaluationIssue(
                "ambiguous_prediction",
                "a prediction marker overlaps multiple required object regions",
                decisive=False,
            )
        matched_regions = {
            matches[0] for matches in prediction_matches if len(matches) == 1
        }
        recall = len(matched_regions) / len(regions)
        passed = recall == 1.0
        return self._result(
            family="coordinate_set",
            task_type="geometry",
            metric="bop_rearrangement_region_recall",
            score=recall,
            passed=passed,
            status="evaluated",
            prediction_normalized=pred,
            ground_truth_normalized={
                "required_region_count": len(regions),
            },
            details={
                **details,
                "comparison": "unique_point_in_required_mask_region",
                "required_region_count": len(regions),
                "matched_region_count": len(matched_regions),
                "unmatched_prediction_count": sum(
                    not matches for matches in prediction_matches
                ),
                "extra_predictions_penalized": False,
                "region_recall": recall,
            },
        )

    def _evaluate_coordinate_common(
        self,
        prediction: object,
        ground_truth: object,
        *,
        choices: Iterable[object] | Mapping[object, object],
        question: str,
        metadata: Mapping[str, Any],
        match_mode: str,
        family: str,
    ) -> TaskTypeEvaluationResultV2:
        del choices
        truth = _parse_points(ground_truth, side="reference").points
        pred = _parse_points(prediction, side="prediction").points
        _validate_points_contract(
            truth,
            metadata=metadata,
            question=question,
            side="reference",
        )
        details = _validate_points_contract(
            pred,
            metadata=metadata,
            question=question,
            side="prediction",
        )
        if len(pred) != len(truth):
            raise _EvaluationIssue(
                "cardinality_mismatch",
                "prediction and reference have different point counts",
            )
        if match_mode == "ordered":
            maximum_error = max(_euclidean_errors(pred, truth), default=0.0)
        else:
            maximum_error = _minimum_bottleneck_error(pred, truth)
        tolerance = _explicit_metadata_float(metadata, "coordinate_tolerance")
        metric = (
            "bop_coordinate_ordered_tolerance"
            if match_mode == "ordered"
            else "bop_coordinate_set_tolerance"
        )
        if tolerance is None:
            return self._result(
                family=family,
                task_type="coordinate",
                metric=metric,
                score=None,
                passed=None,
                status="metric_computed_threshold_unpinned",
                prediction_normalized=pred,
                ground_truth_normalized=truth,
                details={
                    **details,
                    "match_mode": match_mode,
                    "maximum_matching_error_pixels": maximum_error,
                    "threshold_source": "unavailable",
                },
            )
        passed = maximum_error <= tolerance
        return self._result(
            family=family,
            task_type="coordinate",
            metric=metric,
            score=float(passed),
            passed=passed,
            status="evaluated",
            prediction_normalized=pred,
            ground_truth_normalized=truth,
            details={
                **details,
                "match_mode": match_mode,
                "coordinate_tolerance": tolerance,
                "maximum_matching_error_pixels": maximum_error,
                "threshold_source": "coordinate_tolerance",
            },
        )

    def _evaluate_trajectory(
        self,
        prediction: object,
        ground_truth: object,
        *,
        choices: Iterable[object] | Mapping[object, object],
        question: str,
        metadata: Mapping[str, Any],
    ) -> TaskTypeEvaluationResultV2:
        del choices
        truth = _ordered_trajectory_points(
            _parse_points(ground_truth, side="reference"),
            side="reference",
        )
        pred = _ordered_trajectory_points(
            _parse_points(prediction, side="prediction"),
            side="prediction",
        )
        expected_cardinality = metadata.get("trajectory_cardinality", len(truth))
        expected = _metadata_positive_int(
            {"value": expected_cardinality},
            "value",
            len(truth),
        )
        _validate_points_contract(
            truth,
            metadata=metadata,
            question=question,
            side="reference",
            exact_cardinality=expected,
        )
        details = _validate_points_contract(
            pred,
            metadata=metadata,
            question=question,
            side="prediction",
            exact_cardinality=expected,
        )
        if len(pred) != len(truth):
            raise _EvaluationIssue(
                "cardinality_mismatch",
                "trajectory and reference have different waypoint counts",
            )
        errors = _euclidean_errors(pred, truth)
        maximum_error = max(errors, default=0.0)
        mean_error = sum(errors) / len(errors)
        start_error = errors[0]
        endpoint_error = errors[-1]
        waypoint_tolerance = _explicit_metadata_float(
            metadata,
            "trajectory_waypoint_tolerance",
        )
        endpoint_tolerance = _explicit_metadata_float(
            metadata,
            "trajectory_endpoint_tolerance",
        )
        source_region = metadata.get("trajectory_source_region")
        target_region = metadata.get("trajectory_target_region")
        endpoint_regions_available = isinstance(
            source_region,
            Mapping,
        ) and isinstance(target_region, Mapping)
        endpoint_region_success = None
        if endpoint_regions_available:
            endpoint_region_success = (
                _point_in_region(pred[0], source_region)
                and _point_in_region(pred[-1], target_region)
            )
        if waypoint_tolerance is None and endpoint_tolerance is None:
            if endpoint_region_success is None:
                return self._result(
                    family="trajectory",
                    task_type="coordinate",
                    metric="bop_trajectory_endpoint_and_distance",
                    score=None,
                    passed=None,
                    status="metric_computed_threshold_unpinned",
                    prediction_normalized=pred,
                    ground_truth_normalized=truth,
                    details={
                        **details,
                        "match_mode": "ordered_direct_waypoint_alignment",
                        "mean_waypoint_error_pixels": mean_error,
                        "maximum_waypoint_error_pixels": maximum_error,
                        "start_error_pixels": start_error,
                        "endpoint_error_pixels": endpoint_error,
                        "endpoint_regions_available": False,
                        "threshold_source": "unavailable",
                    },
                )
            passed = endpoint_region_success
            metric = "bop_trajectory_endpoint_region_success"
            score = float(passed)
        else:
            passed = (
                (waypoint_tolerance is None or maximum_error <= waypoint_tolerance)
                and (
                    endpoint_tolerance is None
                    or endpoint_error <= endpoint_tolerance
                )
                and (
                    endpoint_region_success is None
                    or endpoint_region_success
                )
            )
            metric = "bop_trajectory_sequence_tolerance"
            score = float(passed)
        return self._result(
            family="trajectory",
            task_type="coordinate",
            metric=metric,
            score=score,
            passed=passed,
            status="evaluated",
            prediction_normalized=pred,
            ground_truth_normalized=truth,
            details={
                **details,
                "match_mode": "ordered_direct_waypoint_alignment",
                "waypoint_tolerance": waypoint_tolerance,
                "endpoint_tolerance": endpoint_tolerance,
                "per_waypoint_errors_pixels": errors,
                "mean_waypoint_error_pixels": mean_error,
                "maximum_waypoint_error_pixels": maximum_error,
                "start_error_pixels": start_error,
                "endpoint_error_pixels": endpoint_error,
                "endpoint_regions_available": endpoint_regions_available,
                "endpoint_region_success": endpoint_region_success,
                "trajectory_direction": "source_to_target",
                "threshold_source": (
                    "explicit_diagnostic_metadata"
                    if waypoint_tolerance is not None
                    or endpoint_tolerance is not None
                    else "endpoint_mask_regions"
                ),
            },
        )

    def _evaluate_bbox(
        self,
        prediction: object,
        ground_truth: object,
        *,
        choices: Iterable[object] | Mapping[object, object],
        question: str,
        metadata: Mapping[str, Any],
    ) -> TaskTypeEvaluationResultV2:
        del choices
        truth = _parse_points(ground_truth, side="reference").points
        pred = _parse_points(prediction, side="prediction").points
        cardinality = _metadata_positive_int(
            metadata,
            "bbox_cardinality",
            self.config.bbox_cardinality,
        )
        _validate_points_contract(
            truth,
            metadata=metadata,
            question=question,
            side="reference",
            exact_cardinality=cardinality,
        )
        details = _validate_points_contract(
            pred,
            metadata=metadata,
            question=question,
            side="prediction",
            exact_cardinality=cardinality,
        )
        pred_bbox = _bbox_from_points(pred, side="prediction")
        truth_bbox = _bbox_from_points(truth, side="reference")
        iou = _bbox_iou(pred_bbox, truth_bbox)
        threshold = _explicit_metadata_float(
            metadata,
            "bbox_iou_threshold",
            maximum=1.0,
        )
        if threshold is None:
            return self._result(
                family="bbox",
                task_type="coordinate",
                metric="bop_bbox_envelope_iou_diagnostic",
                score=iou,
                passed=None,
                status="metric_computed_threshold_unpinned",
                prediction_normalized=pred_bbox,
                ground_truth_normalized=truth_bbox,
                details={
                    **details,
                    "corner_cardinality": cardinality,
                    "bbox_iou": iou,
                    "iou_scope": "axis_aligned_2d_envelope_of_projected_corners",
                    "threshold_source": "unavailable",
                    "uncertainty": (
                        "public sources conflict between typical 2D IoU and "
                        "paper-reported 3D cuboid IoU"
                    ),
                },
            )
        passed = iou >= threshold
        return self._result(
            family="bbox",
            task_type="coordinate",
            metric="bop_bbox_iou",
            score=iou,
            passed=passed,
            status="evaluated",
            prediction_normalized=pred_bbox,
            ground_truth_normalized=truth_bbox,
            details={
                **details,
                "corner_cardinality": cardinality,
                "bbox_iou": iou,
                "bbox_iou_threshold": threshold,
                "iou_scope": "axis_aligned_2d_envelope_of_projected_corners",
                "threshold_source": "explicit_diagnostic_metadata",
            },
        )

    def _evaluate_pose(
        self,
        prediction: object,
        ground_truth: object,
        *,
        choices: Iterable[object] | Mapping[object, object],
        question: str,
        metadata: Mapping[str, Any],
    ) -> TaskTypeEvaluationResultV2:
        del choices, question
        pred = _parse_pose(prediction, metadata=metadata, side="prediction")
        truth = _parse_pose(ground_truth, metadata=metadata, side="reference")
        translation_error = math.sqrt(
            sum(
                (pred_value - truth_value) ** 2
                for pred_value, truth_value in zip(
                    pred.translation,
                    truth.translation,
                )
            )
        )
        rotation_error = _rotation_error_degrees(pred.rotation, truth.rotation)
        translation_tolerance = _metadata_float(
            metadata,
            "pose_translation_tolerance",
            self.config.pose_translation_tolerance,
        )
        rotation_tolerance = _metadata_float(
            metadata,
            "pose_rotation_tolerance_degrees",
            self.config.pose_rotation_tolerance_degrees,
        )
        passed = (
            translation_error <= translation_tolerance
            and rotation_error <= rotation_tolerance
        )
        return self._result(
            family="pose",
            task_type="geometry",
            metric="bop_pose_translation_rotation_tolerance",
            score=float(passed),
            passed=passed,
            status="evaluated",
            prediction_normalized={
                "translation": pred.translation,
                "rotation": pred.rotation,
            },
            ground_truth_normalized={
                "translation": truth.translation,
                "rotation": truth.rotation,
            },
            details={
                "translation_error": translation_error,
                "translation_tolerance": translation_tolerance,
                "rotation_error_degrees": rotation_error,
                "rotation_tolerance_degrees": rotation_tolerance,
                "prediction_rotation_representation": (
                    pred.rotation_representation
                ),
                "reference_rotation_representation": (
                    truth.rotation_representation
                ),
            },
        )

    def _evaluate_relation(
        self,
        prediction: object,
        ground_truth: object | None,
        *,
        choices: Iterable[object] | Mapping[object, object],
        question: str,
        metadata: Mapping[str, Any],
    ) -> TaskTypeEvaluationResultV2:
        del question
        prediction_yes_no = _yes_no(prediction, choices)
        truth_yes_no = _yes_no(ground_truth, choices)
        relation_geometry = metadata.get("relation_geometry")
        if isinstance(relation_geometry, Mapping):
            expected, relation_name, geometry_details = _evaluate_relation_geometry(
                relation_geometry,
                config=self.config,
                metadata=metadata,
            )
            if truth_yes_no is not None and truth_yes_no != expected:
                return self._result(
                    family="relation",
                    task_type="geometry",
                    metric="bop_relation_geometry",
                    score=None,
                    passed=None,
                    status="ambiguous_reference",
                    prediction_normalized=prediction_yes_no,
                    ground_truth_normalized=truth_yes_no,
                    details={
                        **geometry_details,
                        "reason": (
                            "reference label conflicts with declared relation geometry"
                        ),
                    },
                )
            if prediction_yes_no is None:
                raise _EvaluationIssue(
                    "parse_failed",
                    "relation-geometry prediction must resolve to yes or no",
                )
            passed = prediction_yes_no == expected
            return self._result(
                family="relation",
                task_type="geometry",
                metric="bop_relation_geometry",
                score=float(passed),
                passed=passed,
                status="evaluated",
                prediction_normalized=prediction_yes_no,
                ground_truth_normalized=expected,
                details={
                    **geometry_details,
                    "relation": relation_name,
                    "comparison": "derived_geometry_boolean",
                },
            )
        if prediction_yes_no is not None or truth_yes_no is not None:
            if prediction_yes_no is None:
                raise _EvaluationIssue(
                    "parse_failed",
                    "prediction does not resolve to yes or no",
                )
            if truth_yes_no is None:
                raise _EvaluationIssue(
                    "parse_failed",
                    "reference does not resolve to yes or no",
                    side="reference",
                    decisive=False,
                )
            passed = prediction_yes_no == truth_yes_no
            return self._result(
                family="relation",
                task_type="geometry",
                metric="bop_relation_label",
                score=float(passed),
                passed=passed,
                status="evaluated",
                prediction_normalized=prediction_yes_no,
                ground_truth_normalized=truth_yes_no,
                details={"comparison": "normalized_yes_no"},
            )
        pred_relation = _parse_relation(prediction, side="prediction")
        truth_relation = _parse_relation(ground_truth, side="reference")
        if pred_relation is None:
            raise _EvaluationIssue(
                "parse_failed",
                "prediction has no supported spatial relation",
            )
        if truth_relation is None:
            raise _EvaluationIssue(
                "parse_failed",
                "reference has no supported spatial relation",
                side="reference",
                decisive=False,
            )
        passed = _relations_match(pred_relation, truth_relation)
        return self._result(
            family="relation",
            task_type="geometry",
            metric="bop_relation_semantic",
            score=float(passed),
            passed=passed,
            status="evaluated",
            prediction_normalized={
                "relation": pred_relation.name,
                "subject": pred_relation.subject,
                "object": pred_relation.object,
            },
            ground_truth_normalized={
                "relation": truth_relation.name,
                "subject": truth_relation.subject,
                "object": truth_relation.object,
            },
            details={"comparison": "relation_with_inverse_entity_order"},
        )


__all__ = [
    "ANALYSIS_ONLY",
    "BOPASKMetricConfigV1",
    "BOPASKEvaluatorV1",
    "EVALUATOR_VERSION",
]
