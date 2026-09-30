"""Offline, task-aware evaluator routing for analysis and future builders.

This module is intentionally not imported by the production runner. It performs
deterministic local parsing and comparison only; it has no API, benchmark
execution, data mutation, or primitive-registry dependency.
"""

from __future__ import annotations

import ast
import json
import math
import re
import string
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable


TASK_TYPES = {
    "choice",
    "yes_no",
    "numeric",
    "coordinate",
    "geometry",
    "text",
}
ANALYSIS_ONLY = True

_NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_NUMBER_RE = re.compile(_NUMBER_PATTERN)
_OPTION_PREFIX_RE = re.compile(
    r"^\s*(?:option\s+)?[\(\[]?([A-Za-z]|\d+)[\)\]]?"
    r"(?:\s*[\.\:\)\-]\s*|\s*$)",
    flags=re.IGNORECASE,
)
_ANSWER_LABEL_RE = re.compile(
    r"^\s*(?:the\s+)?(?:answer|option|choice)\s*(?:is|:)?\s*"
    r"[\(\[]?([A-Za-z]|\d+)[\)\]]?[\.\s]*$",
    flags=re.IGNORECASE,
)


CoordinateRegionValidator = Callable[
    [tuple[tuple[float, float], ...], dict[str, Any]],
    tuple[bool, float, dict[str, Any]],
]


@dataclass(frozen=True)
class TaskMetricConfigV2:
    """Validated defaults for deterministic v2 diagnostic metrics."""

    numeric_abs_tolerance: float = 1e-3
    numeric_rel_tolerance: float = 1e-6
    coordinate_minimum: float = 0.0
    coordinate_maximum: float = 1.0
    coordinate_min_cardinality: int = 1
    coordinate_max_cardinality: int = 100
    coordinate_tolerance: float = 0.03
    geometry_tolerance: float = 0.05
    geometry_rel_tolerance: float = 1e-6
    text_ignore_articles: bool = False

    def validate(self) -> None:
        numeric_values = (
            self.numeric_abs_tolerance,
            self.numeric_rel_tolerance,
            self.coordinate_tolerance,
            self.geometry_tolerance,
            self.geometry_rel_tolerance,
        )
        if any(not math.isfinite(value) or value < 0 for value in numeric_values):
            raise ValueError("all evaluator tolerances must be finite and non-negative")
        if (
            not math.isfinite(self.coordinate_minimum)
            or not math.isfinite(self.coordinate_maximum)
            or self.coordinate_minimum >= self.coordinate_maximum
        ):
            raise ValueError("coordinate range is invalid")
        if self.coordinate_min_cardinality < 1:
            raise ValueError("coordinate minimum cardinality must be positive")
        if self.coordinate_max_cardinality < self.coordinate_min_cardinality:
            raise ValueError("coordinate cardinality range is invalid")


@dataclass(frozen=True)
class TaskTypeEvaluationResultV2:
    """A v1-compatible, three-valued diagnostic result."""

    task_type: str
    metric: str
    score: float | None
    passed: bool | None
    status: str
    prediction_normalized: Any
    ground_truth_normalized: Any
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "metric": self.metric,
            "score": self.score,
            "passed": self.passed,
            "status": self.status,
            "prediction_normalized": self.prediction_normalized,
            "ground_truth_normalized": self.ground_truth_normalized,
            "details": self.details,
        }


@dataclass(frozen=True)
class _Quantity:
    value: float
    dimension: str
    canonical_unit: str | None


_UNIT_ALIASES: dict[str, tuple[str, float, str | None]] = {
    "millimeter": ("length", 1e-3, "m"),
    "millimeters": ("length", 1e-3, "m"),
    "millimetre": ("length", 1e-3, "m"),
    "millimetres": ("length", 1e-3, "m"),
    "mm": ("length", 1e-3, "m"),
    "centimeter": ("length", 1e-2, "m"),
    "centimeters": ("length", 1e-2, "m"),
    "centimetre": ("length", 1e-2, "m"),
    "centimetres": ("length", 1e-2, "m"),
    "cm": ("length", 1e-2, "m"),
    "kilometer": ("length", 1e3, "m"),
    "kilometers": ("length", 1e3, "m"),
    "kilometre": ("length", 1e3, "m"),
    "kilometres": ("length", 1e3, "m"),
    "km": ("length", 1e3, "m"),
    "meter": ("length", 1.0, "m"),
    "meters": ("length", 1.0, "m"),
    "metre": ("length", 1.0, "m"),
    "metres": ("length", 1.0, "m"),
    "m": ("length", 1.0, "m"),
    "millisecond": ("time", 1e-3, "s"),
    "milliseconds": ("time", 1e-3, "s"),
    "ms": ("time", 1e-3, "s"),
    "second": ("time", 1.0, "s"),
    "seconds": ("time", 1.0, "s"),
    "sec": ("time", 1.0, "s"),
    "secs": ("time", 1.0, "s"),
    "s": ("time", 1.0, "s"),
    "minute": ("time", 60.0, "s"),
    "minutes": ("time", 60.0, "s"),
    "min": ("time", 60.0, "s"),
    "mins": ("time", 60.0, "s"),
    "hour": ("time", 3600.0, "s"),
    "hours": ("time", 3600.0, "s"),
    "hr": ("time", 3600.0, "s"),
    "hrs": ("time", 3600.0, "s"),
    "radian": ("angle", 1.0, "rad"),
    "radians": ("angle", 1.0, "rad"),
    "rad": ("angle", 1.0, "rad"),
    "degree": ("angle", math.pi / 180.0, "rad"),
    "degrees": ("angle", math.pi / 180.0, "rad"),
    "deg": ("angle", math.pi / 180.0, "rad"),
    "\N{DEGREE SIGN}": ("angle", math.pi / 180.0, "rad"),
    "pixel": ("pixel", 1.0, "px"),
    "pixels": ("pixel", 1.0, "px"),
    "px": ("pixel", 1.0, "px"),
    "kilogram": ("mass", 1.0, "kg"),
    "kilograms": ("mass", 1.0, "kg"),
    "kg": ("mass", 1.0, "kg"),
    "gram": ("mass", 1e-3, "kg"),
    "grams": ("mass", 1e-3, "kg"),
    "g": ("mass", 1e-3, "kg"),
    "percent": ("dimensionless", 1e-2, None),
    "percentage": ("dimensionless", 1e-2, None),
    "pct": ("dimensionless", 1e-2, None),
    "%": ("dimensionless", 1e-2, None),
}
_SORTED_UNIT_ALIASES = tuple(
    sorted(_UNIT_ALIASES, key=lambda item: (-len(item), item))
)


_RELATION_ALIASES = {
    "left": "left_of",
    "left of": "left_of",
    "to the left of": "left_of",
    "right": "right_of",
    "right of": "right_of",
    "to the right of": "right_of",
    "front": "front_of",
    "front of": "front_of",
    "in front": "front_of",
    "in front of": "front_of",
    "behind": "behind",
    "back of": "behind",
    "above": "above",
    "over": "above",
    "below": "below",
    "under": "below",
    "near": "near",
    "nearer": "near",
    "close": "near",
    "closer": "near",
    "far": "far",
    "farther": "far",
    "further": "far",
    "touching": "touching",
    "in contact with": "touching",
    "overlapping": "overlap",
    "overlap": "overlap",
    "intersecting": "intersect",
    "intersect": "intersect",
    "inside": "inside",
    "within": "inside",
    "contained in": "inside",
    "contains": "contains",
    "around": "contains",
    "disjoint": "disjoint",
    "separate": "disjoint",
    "aligned": "aligned",
    "same position": "same_position",
}
_SORTED_RELATION_ALIASES = tuple(
    sorted(_RELATION_ALIASES, key=lambda item: (-len(item), item))
)


def _is_blank(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _semantic_text(value: object, *, ignore_articles: bool = False) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.replace("&", " and ")
    text = re.sub(r"[\W_]+", " ", text, flags=re.UNICODE)
    tokens = text.split()
    if ignore_articles:
        tokens = [token for token in tokens if token not in {"a", "an", "the"}]
    return " ".join(tokens)


def _strip_option_prefix(value: object) -> str:
    text = "" if value is None else str(value).strip()
    match = _OPTION_PREFIX_RE.match(text)
    if match:
        return text[match.end() :].strip()
    return text


def _choice_entries(choices: Iterable[object] | Mapping[object, object]) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    if isinstance(choices, Mapping):
        source: Iterable[object] = tuple(choices.items())
    elif isinstance(choices, str):
        source = (choices,)
    else:
        source = choices
    source = tuple(source)
    if len(source) == 1 and isinstance(source[0], str):
        compact = source[0].strip()
        markers = tuple(
            re.finditer(
                r"(?:^|(?<=[.;]))\s*([A-Za-z]|\d+)\s*[\.\:\)\-]\s*",
                compact,
            )
        )
        labels = tuple(marker.group(1).upper() for marker in markers)
        expected_labels = tuple(
            string.ascii_uppercase[index] for index in range(len(markers))
        )
        if len(markers) > 1 and labels == expected_labels:
            split_entries: list[tuple[str, str]] = []
            for marker_index, marker in enumerate(markers):
                end = (
                    markers[marker_index + 1].start()
                    if marker_index + 1 < len(markers)
                    else len(compact)
                )
                text = compact[marker.end() : end].strip().rstrip(";").strip()
                if text:
                    split_entries.append((marker.group(1).upper(), text))
            if len(split_entries) == len(markers):
                return split_entries
    for index, raw in enumerate(source):
        fallback_label = string.ascii_uppercase[index] if index < 26 else str(index + 1)
        if isinstance(raw, Mapping):
            label = str(raw.get("label") or raw.get("key") or fallback_label).upper()
            text = str(raw.get("text") or raw.get("value") or "").strip()
        elif (
            isinstance(raw, Sequence)
            and not isinstance(raw, str)
            and len(raw) == 2
        ):
            label, text = str(raw[0]).upper(), str(raw[1]).strip()
        else:
            raw_text = str(raw).strip()
            match = _OPTION_PREFIX_RE.match(raw_text)
            if match and raw_text[match.end() :].strip():
                label = match.group(1).upper()
                text = raw_text[match.end() :].strip()
            else:
                label, text = fallback_label, raw_text
        entries.append((label, text))
    return entries


def _choice_map(
    choices: Iterable[object] | Mapping[object, object],
) -> dict[str, str]:
    return dict(_choice_entries(choices))


def _extract_option_label(value: object, allowed: set[str]) -> str | None:
    text = "" if value is None else str(value).strip()
    boxed = re.fullmatch(r"\\boxed\{\s*([A-Za-z]|\d+)\s*\}", text)
    if boxed and boxed.group(1).upper() in allowed:
        return boxed.group(1).upper()
    answer = _ANSWER_LABEL_RE.match(text)
    if answer and answer.group(1).upper() in allowed:
        return answer.group(1).upper()
    prefix = _OPTION_PREFIX_RE.match(text)
    if prefix and prefix.group(1).upper() in allowed:
        return prefix.group(1).upper()
    return None


def _resolve_option(
    value: object,
    choices: Iterable[object] | Mapping[object, object],
) -> tuple[str | None, str]:
    mapping = _choice_map(choices)
    label = _extract_option_label(value, set(mapping))
    if label is not None:
        return label, mapping[label]
    text = "" if value is None else str(value).strip()
    prefix = _OPTION_PREFIX_RE.match(text)
    if prefix and prefix.group(1).isalpha():
        return None, text[prefix.end() :].strip()
    return None, text


def _canonical_yes_no(value: object) -> str | None:
    text = _semantic_text(_strip_option_prefix(value))
    positive = {
        "yes",
        "y",
        "true",
        "affirmative",
        "correct",
        "possible",
        "can",
    }
    negative = {
        "no",
        "n",
        "false",
        "negative",
        "incorrect",
        "impossible",
        "cannot",
        "cant",
    }
    if text in positive:
        return "yes"
    if text in negative:
        return "no"
    first = text.split(" ", 1)[0] if text else ""
    if first in positive:
        return "yes"
    if first in negative:
        return "no"
    return None


def _resolved_yes_no(
    value: object,
    choices: Iterable[object] | Mapping[object, object],
) -> tuple[str | None, str | None]:
    label, option_text = _resolve_option(value, choices)
    return _canonical_yes_no(option_text), label


def _unit_from_suffix(suffix: str, *, strip_closing: bool = False) -> str | None:
    candidate = suffix.lstrip()
    if strip_closing:
        candidate = re.sub(r"^[\)\]\}>,;:\s]+", "", candidate)
    lowered = candidate.casefold()
    for alias in _SORTED_UNIT_ALIASES:
        if not lowered.startswith(alias):
            continue
        tail = lowered[len(alias) :]
        if alias[-1:].isalpha() and tail[:1].isalpha():
            continue
        return alias
    return None


def _unit_spec(unit: object | None) -> tuple[str, float, str | None]:
    if unit is None or not str(unit).strip():
        return ("dimensionless", 1.0, None)
    token = str(unit).strip().casefold()
    spec = _UNIT_ALIASES.get(token)
    if spec is None:
        raise ValueError(f"unsupported unit hint: {unit}")
    return spec


def _quantity_from_number(number: float, unit: object | None) -> _Quantity:
    dimension, factor, canonical_unit = _unit_spec(unit)
    normalized = number * factor
    if not math.isfinite(normalized):
        raise ValueError("quantity is not finite")
    return _Quantity(normalized, dimension, canonical_unit)


def _parse_quantity(value: object, *, default_unit: object | None = None) -> _Quantity | None:
    text = "" if value is None else str(value)
    match = _NUMBER_RE.search(text)
    if match is None:
        return None
    try:
        number = float(match.group(0))
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    unit = _unit_from_suffix(text[match.end() :]) or default_unit
    return _quantity_from_number(number, unit)


def _unit_hint_from_question(question: object) -> str | None:
    text = unicodedata.normalize("NFKC", str(question or "")).casefold()
    patterns = (
        r"\(\s*in\s+([a-z%°]+)\s*\)",
        r"\b(?:answer|measure|measured|distance|length|time|angle)\s+in\s+([a-z%°]+)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match and match.group(1) in _UNIT_ALIASES:
            return match.group(1)
    return None


def _parse_quantity_vector(
    value: object,
    *,
    default_unit: object | None = None,
) -> tuple[_Quantity, ...] | None:
    text = "" if value is None else str(value)
    matches = tuple(_NUMBER_RE.finditer(text))
    if not matches:
        return None
    global_unit = _unit_from_suffix(
        text[matches[-1].end() :],
        strip_closing=True,
    )
    quantities: list[_Quantity] = []
    for match in matches:
        number = float(match.group(0))
        if not math.isfinite(number):
            return None
        unit = _unit_from_suffix(text[match.end() :]) or global_unit or default_unit
        quantities.append(_quantity_from_number(number, unit))
    return tuple(quantities)


def _coerce_coordinate_pairs(value: object) -> tuple[tuple[float, float], ...] | None:
    if isinstance(value, Mapping):
        if "x" not in value or "y" not in value:
            return None
        source: object = ((value["x"], value["y"]),)
    else:
        source = value
    if not isinstance(source, Sequence) or isinstance(source, (str, bytes)):
        return None
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
            return None
        source = tuple(zip(source[::2], source[1::2]))
    pairs: list[tuple[float, float]] = []
    for item in source:
        if isinstance(item, Mapping):
            if "x" not in item or "y" not in item:
                return None
            raw_x, raw_y = item["x"], item["y"]
        elif (
            isinstance(item, Sequence)
            and not isinstance(item, (str, bytes))
            and len(item) == 2
        ):
            raw_x, raw_y = item
        else:
            return None
        if isinstance(raw_x, bool) or isinstance(raw_y, bool):
            return None
        try:
            x, y = float(raw_x), float(raw_y)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        pairs.append((x, y))
    return tuple(pairs)


def _coordinate_pairs(value: object) -> tuple[tuple[float, float], ...] | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return _coerce_coordinate_pairs(value)
    text = value.strip()
    xml_pairs: list[tuple[str, str]] = []
    for tag in re.findall(r"<points?\b[^>]*>", text, flags=re.IGNORECASE):
        attributes = {
            key.casefold(): raw
            for key, _, raw in re.findall(
                r"([A-Za-z_][\w\-]*)\s*=\s*([\"'])\s*([^\"']+?)\s*\2",
                tag,
            )
        }
        if "x" in attributes and "y" in attributes:
            xml_pairs.append((attributes["x"], attributes["y"]))
    if xml_pairs:
        return _coerce_coordinate_pairs(xml_pairs)
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(text)
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            continue
        pairs = _coerce_coordinate_pairs(parsed)
        if pairs is not None:
            return pairs
    pair_matches = re.findall(
        rf"[\(\[]\s*({_NUMBER_PATTERN})\s*,\s*({_NUMBER_PATTERN})\s*[\)\]]",
        text,
    )
    if pair_matches:
        return _coerce_coordinate_pairs(pair_matches)
    named_matches = re.findall(
        rf"\bx\s*=\s*({_NUMBER_PATTERN})\D+\by\s*=\s*({_NUMBER_PATTERN})",
        text,
        flags=re.IGNORECASE,
    )
    return _coerce_coordinate_pairs(named_matches) if named_matches else None


def _finite_pair(value: object) -> tuple[float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if len(value) != 2:
        return None
    try:
        pair = (float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in pair) or pair[0] >= pair[1]:
        return None
    return pair


def _coordinate_bounds(
    metadata: Mapping[str, Any],
    question: str,
    config: TaskMetricConfigV2,
) -> tuple[tuple[float, float], tuple[float, float]]:
    default = (config.coordinate_minimum, config.coordinate_maximum)
    x_bounds = default
    y_bounds = default
    raw_bounds = metadata.get("coordinate_bounds")
    if isinstance(raw_bounds, Mapping):
        x_bounds = _finite_pair(raw_bounds.get("x")) or x_bounds
        y_bounds = _finite_pair(raw_bounds.get("y")) or y_bounds
    elif isinstance(raw_bounds, Sequence) and not isinstance(raw_bounds, (str, bytes)):
        if len(raw_bounds) == 4:
            x_bounds = _finite_pair(raw_bounds[:2]) or x_bounds
            y_bounds = _finite_pair(raw_bounds[2:]) or y_bounds
    x_bounds = _finite_pair(metadata.get("x_range")) or x_bounds
    y_bounds = _finite_pair(metadata.get("y_range")) or y_bounds
    width = metadata.get("image_width")
    height = metadata.get("image_height")
    try:
        if width is not None and math.isfinite(float(width)) and float(width) > 0:
            x_bounds = (0.0, float(width))
        if height is not None and math.isfinite(float(height)) and float(height) > 0:
            y_bounds = (0.0, float(height))
    except (TypeError, ValueError):
        pass
    patterns = {
        "x": (
            rf"\bx\s+(?:should\s+be\s+)?between\s*[\(\[]\s*"
            rf"({_NUMBER_PATTERN})\s*,\s*({_NUMBER_PATTERN})\s*[\)\]]",
            rf"({_NUMBER_PATTERN})\s*<=?\s*x\s*<=?\s*({_NUMBER_PATTERN})",
        ),
        "y": (
            rf"\by\s+(?:should\s+be\s+)?between\s*[\(\[]\s*"
            rf"({_NUMBER_PATTERN})\s*,\s*({_NUMBER_PATTERN})\s*[\)\]]",
            rf"({_NUMBER_PATTERN})\s*<=?\s*y\s*<=?\s*({_NUMBER_PATTERN})",
        ),
    }
    for axis, axis_patterns in patterns.items():
        for pattern in axis_patterns:
            match = re.search(pattern, question, flags=re.IGNORECASE)
            if match:
                parsed = _finite_pair((match.group(1), match.group(2)))
                if parsed is not None:
                    if axis == "x":
                        x_bounds = parsed
                    else:
                        y_bounds = parsed
                break
    return x_bounds, y_bounds


def _coordinate_errors(
    prediction: tuple[tuple[float, float], ...],
    ground_truth: tuple[tuple[float, float], ...],
    *,
    mode: str,
) -> tuple[float, ...]:
    if mode == "set":
        remaining = list(ground_truth)
        errors: list[float] = []
        for point in prediction:
            distances = [math.dist(point, candidate) for candidate in remaining]
            index = min(range(len(distances)), key=distances.__getitem__)
            errors.append(distances[index])
            remaining.pop(index)
        return tuple(errors)
    return tuple(
        math.dist(prediction_point, truth_point)
        for prediction_point, truth_point in zip(prediction, ground_truth)
    )


def _canonical_relation(value: object) -> str | None:
    text = _semantic_text(value)
    if not text:
        return None
    direct = _RELATION_ALIASES.get(text)
    if direct is not None:
        return direct
    for alias in _SORTED_RELATION_ALIASES:
        if re.search(rf"\b{re.escape(alias)}\b", text):
            return _RELATION_ALIASES[alias]
    return None


def _metadata_tolerance(
    metadata: Mapping[str, Any],
    key: str,
    default: float,
) -> float:
    raw = metadata.get(key, default)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{key} must be finite and non-negative")
    return value


def _canonicalize_alias(
    normalized: str,
    aliases: object,
) -> str:
    if not isinstance(aliases, Mapping):
        return normalized
    for canonical, raw_aliases in aliases.items():
        canonical_norm = _semantic_text(canonical)
        if isinstance(raw_aliases, Sequence) and not isinstance(raw_aliases, str):
            group = {_semantic_text(item) for item in raw_aliases}
        else:
            group = {_semantic_text(raw_aliases)}
        group.add(canonical_norm)
        if normalized in group:
            return canonical_norm
    return normalized


class TaskTypeEvaluatorRouterV2:
    """Route submitted answers to offline v2 diagnostic metrics."""

    def __init__(
        self,
        config: TaskMetricConfigV2 | None = None,
        *,
        coordinate_region_validator: CoordinateRegionValidator | None = None,
        coordinate_validator: CoordinateRegionValidator | None = None,
    ) -> None:
        if (
            coordinate_region_validator is not None
            and coordinate_validator is not None
            and coordinate_region_validator is not coordinate_validator
        ):
            raise ValueError("provide only one coordinate validator")
        self.config = config or TaskMetricConfigV2()
        self.config.validate()
        self.coordinate_region_validator = (
            coordinate_region_validator or coordinate_validator
        )

    def infer_task_type(
        self,
        *,
        prediction: object,
        ground_truth: object,
        answer_type: str,
        evaluator: str,
        choices: Iterable[object] | Mapping[object, object] = (),
        question: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        metadata = metadata or {}
        explicit = str(
            metadata.get("metric_task_type")
            or metadata.get("evaluator_task_type")
            or metadata.get("task_type")
            or ""
        ).casefold()
        explicit = {"yes/no": "yes_no", "yesno": "yes_no"}.get(explicit, explicit)
        if explicit in TASK_TYPES:
            return explicit
        evaluator_name = str(evaluator or "").casefold()
        answer_kind = str(answer_type or "").casefold()
        if evaluator_name in {
            "coordinate",
            "coordinate_tolerance",
            "coordinate_region",
        }:
            return "coordinate"
        if evaluator_name in {
            "geometry",
            "geometry_tolerance",
            "geometry_relation",
        }:
            return "geometry"
        if answer_kind == "numeric" or evaluator_name in {
            "numeric",
            "numeric_exact",
            "numeric_tolerance",
        }:
            return "numeric"
        if answer_kind == "multiple_choice" or evaluator_name in {
            "choice",
            "choice_exact",
            "semantic_choice",
        }:
            return "choice"
        coordinate_hint = any(
            token in str(question).casefold()
            for token in (
                "coordinate",
                "list of tuples",
                "normalized pixel",
                "point",
            )
        )
        if coordinate_hint and (
            _coordinate_pairs(prediction) is not None
            or _coordinate_pairs(ground_truth) is not None
        ):
            return "coordinate"
        ground_truth_yes_no, _ = _resolved_yes_no(ground_truth, choices)
        choice_values = [
            _canonical_yes_no(option_text)
            for _, option_text in _choice_entries(choices)
        ]
        if ground_truth_yes_no is not None or (
            choice_values
            and all(item is not None for item in choice_values)
            and {"yes", "no"}.issubset(set(choice_values))
        ):
            return "yes_no"
        question_type = " ".join(
            str(metadata.get(key) or "").casefold()
            for key in ("question_type", "question_subtype")
        )
        if any(
            token in question_type
            for token in (
                "geometry",
                "pose",
                "distance",
                "depth",
                "trajectory",
                "spatial_reasoning",
                "relative_position",
            )
        ):
            return "geometry"
        return "text"

    def evaluate(
        self,
        prediction: object,
        ground_truth: object | None,
        *,
        answer_type: str = "short_text",
        evaluator: str = "normalized_match",
        benchmark: str = "",
        choices: Iterable[object] | Mapping[object, object] = (),
        question: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> TaskTypeEvaluationResultV2:
        if isinstance(choices, Mapping) or isinstance(choices, str):
            routed_choices: Iterable[object] | Mapping[object, object] = choices
        else:
            routed_choices = tuple(choices)
        routed_metadata = dict(metadata or {})
        benchmark_name = str(
            benchmark
            or routed_metadata.get("benchmark")
            or routed_metadata.get("benchmark_name")
            or ""
        )
        benchmark_marker = re.sub(
            r"[^a-z0-9]+",
            "",
            benchmark_name.casefold(),
        )
        evaluator_marker = str(evaluator or "").casefold().replace("-", "_")
        if benchmark_marker == "bopask" or evaluator_marker == "bop_ask_v1":
            # Local import keeps the generic router independent while allowing
            # an explicit, analysis-only BOP-ASK extension.
            from .bop_ask_v1 import BOPASKEvaluatorV1

            return BOPASKEvaluatorV1().evaluate(
                prediction,
                ground_truth,
                answer_type=answer_type,
                evaluator=evaluator,
                choices=routed_choices,
                question=question,
                metadata=routed_metadata,
            )
        task_type = self.infer_task_type(
            prediction=prediction,
            ground_truth=ground_truth,
            answer_type=answer_type,
            evaluator=evaluator,
            choices=routed_choices,
            question=question,
            metadata=metadata,
        )
        if _is_blank(prediction):
            return TaskTypeEvaluationResultV2(
                task_type=task_type,
                metric="non_empty",
                score=0.0,
                passed=False,
                status="invalid_prediction",
                prediction_normalized="",
                ground_truth_normalized=None,
                details={"reason": "empty_prediction"},
            )
        validator_can_evaluate = (
            task_type == "coordinate"
            and self.coordinate_region_validator is not None
        )
        if _is_blank(ground_truth) and not validator_can_evaluate:
            return TaskTypeEvaluationResultV2(
                task_type=task_type,
                metric="non_empty",
                score=None,
                passed=None,
                status="reference_unavailable",
                prediction_normalized=str(prediction).strip(),
                ground_truth_normalized=None,
            )
        routed_metadata["_router_question"] = question
        method = getattr(self, f"_evaluate_{task_type}")
        return method(
            prediction,
            ground_truth,
            choices=routed_choices,
            metadata=routed_metadata,
        )

    def _evaluate_choice(
        self,
        prediction: object,
        ground_truth: object,
        *,
        choices: Iterable[object] | Mapping[object, object],
        metadata: dict[str, Any],
    ) -> TaskTypeEvaluationResultV2:
        pred_label, pred_value = _resolve_option(prediction, choices)
        truth_label, truth_value = _resolve_option(ground_truth, choices)
        ignore_articles = bool(
            metadata.get("text_ignore_articles", self.config.text_ignore_articles)
        )
        pred_norm = _semantic_text(pred_value, ignore_articles=ignore_articles)
        truth_norm = _semantic_text(truth_value, ignore_articles=ignore_articles)
        passed = bool(pred_norm) and pred_norm == truth_norm
        return TaskTypeEvaluationResultV2(
            task_type="choice",
            metric="semantic_choice",
            score=float(passed),
            passed=passed,
            status="evaluated",
            prediction_normalized=pred_norm,
            ground_truth_normalized=truth_norm,
            details={
                "prediction_label": pred_label,
                "ground_truth_label": truth_label,
                "comparison": "resolved_option_text",
            },
        )

    def _evaluate_yes_no(
        self,
        prediction: object,
        ground_truth: object,
        *,
        choices: Iterable[object] | Mapping[object, object],
        metadata: dict[str, Any],
    ) -> TaskTypeEvaluationResultV2:
        del metadata
        pred, pred_label = _resolved_yes_no(prediction, choices)
        truth, truth_label = _resolved_yes_no(ground_truth, choices)
        if pred is None or truth is None:
            return TaskTypeEvaluationResultV2(
                task_type="yes_no",
                metric="semantic_yes_no",
                score=0.0,
                passed=False,
                status="parse_failed",
                prediction_normalized=pred,
                ground_truth_normalized=truth,
                details={
                    "prediction_label": pred_label,
                    "ground_truth_label": truth_label,
                },
            )
        passed = pred == truth
        return TaskTypeEvaluationResultV2(
            task_type="yes_no",
            metric="semantic_yes_no",
            score=float(passed),
            passed=passed,
            status="evaluated",
            prediction_normalized=pred,
            ground_truth_normalized=truth,
            details={
                "prediction_label": pred_label,
                "ground_truth_label": truth_label,
            },
        )

    def _evaluate_numeric(
        self,
        prediction: object,
        ground_truth: object,
        *,
        choices: Iterable[object] | Mapping[object, object],
        metadata: dict[str, Any],
    ) -> TaskTypeEvaluationResultV2:
        del choices
        common_unit = metadata.get(
            "numeric_unit",
            metadata.get(
                "unit",
                _unit_hint_from_question(metadata.get("_router_question")),
            ),
        )
        pred = _parse_quantity(
            prediction,
            default_unit=metadata.get("prediction_unit", common_unit),
        )
        truth = _parse_quantity(
            ground_truth,
            default_unit=metadata.get(
                "ground_truth_unit",
                metadata.get("reference_unit", common_unit),
            ),
        )
        if pred is None or truth is None:
            return TaskTypeEvaluationResultV2(
                task_type="numeric",
                metric="numeric_unit_tolerance",
                score=0.0,
                passed=False,
                status="parse_failed",
                prediction_normalized=None if pred is None else pred.value,
                ground_truth_normalized=None if truth is None else truth.value,
            )
        if pred.dimension != truth.dimension:
            return TaskTypeEvaluationResultV2(
                task_type="numeric",
                metric="numeric_unit_tolerance",
                score=0.0,
                passed=False,
                status="unit_mismatch",
                prediction_normalized=pred.value,
                ground_truth_normalized=truth.value,
                details={
                    "prediction_dimension": pred.dimension,
                    "ground_truth_dimension": truth.dimension,
                },
            )
        abs_tolerance = _metadata_tolerance(
            metadata,
            "numeric_abs_tolerance",
            self.config.numeric_abs_tolerance,
        )
        rel_tolerance = _metadata_tolerance(
            metadata,
            "numeric_rel_tolerance",
            self.config.numeric_rel_tolerance,
        )
        absolute_error = abs(pred.value - truth.value)
        passed = math.isclose(
            pred.value,
            truth.value,
            rel_tol=rel_tolerance,
            abs_tol=abs_tolerance,
        )
        return TaskTypeEvaluationResultV2(
            task_type="numeric",
            metric="numeric_unit_tolerance",
            score=float(passed),
            passed=passed,
            status="evaluated",
            prediction_normalized=pred.value,
            ground_truth_normalized=truth.value,
            details={
                "canonical_unit": pred.canonical_unit,
                "absolute_error": absolute_error,
                "absolute_tolerance": abs_tolerance,
                "relative_tolerance": rel_tolerance,
            },
        )

    def _evaluate_coordinate(
        self,
        prediction: object,
        ground_truth: object | None,
        *,
        choices: Iterable[object] | Mapping[object, object],
        metadata: dict[str, Any],
    ) -> TaskTypeEvaluationResultV2:
        del choices
        pred = _coordinate_pairs(prediction)
        truth = _coordinate_pairs(ground_truth)
        if pred is None:
            return TaskTypeEvaluationResultV2(
                task_type="coordinate",
                metric="coordinate_tolerance",
                score=0.0,
                passed=False,
                status="parse_failed",
                prediction_normalized=None,
                ground_truth_normalized=truth,
            )
        minimum = int(
            metadata.get(
                "coordinate_min_cardinality",
                self.config.coordinate_min_cardinality,
            )
        )
        maximum = int(
            metadata.get(
                "coordinate_max_cardinality",
                self.config.coordinate_max_cardinality,
            )
        )
        exact = metadata.get(
            "coordinate_cardinality",
            metadata.get("expected_cardinality"),
        )
        if minimum < 1 or maximum < minimum:
            raise ValueError("coordinate cardinality metadata is invalid")
        count_valid = minimum <= len(pred) <= maximum
        if exact is not None:
            exact_count = int(exact)
            if exact_count < 1:
                raise ValueError("coordinate cardinality must be positive")
            count_valid = count_valid and len(pred) == exact_count
        question = str(metadata.get("_router_question") or "")
        x_bounds, y_bounds = _coordinate_bounds(metadata, question, self.config)
        range_valid = all(
            x_bounds[0] <= x <= x_bounds[1]
            and y_bounds[0] <= y <= y_bounds[1]
            for x, y in pred
        )
        details = {
            "cardinality": len(pred),
            "cardinality_valid": count_valid,
            "range_valid": range_valid,
            "x_range": x_bounds,
            "y_range": y_bounds,
        }
        if not count_valid or not range_valid:
            return TaskTypeEvaluationResultV2(
                task_type="coordinate",
                metric="coordinate_contract",
                score=0.0,
                passed=False,
                status="invalid_coordinate_contract",
                prediction_normalized=pred,
                ground_truth_normalized=truth,
                details=details,
            )
        if self.coordinate_region_validator is not None:
            passed, score, validation_details = self.coordinate_region_validator(
                pred,
                metadata,
            )
            numeric_score = float(score)
            if not math.isfinite(numeric_score):
                raise ValueError("coordinate validator score must be finite")
            return TaskTypeEvaluationResultV2(
                task_type="coordinate",
                metric="coordinate_region",
                score=numeric_score,
                passed=bool(passed),
                status="evaluated",
                prediction_normalized=pred,
                ground_truth_normalized=None,
                details={**details, **dict(validation_details)},
            )
        reference_mode = str(
            metadata.get(
                "coordinate_match_mode",
                metadata.get("coordinate_reference_mode", "ordered"),
            )
        ).casefold()
        reference_mode = {
            "reference_set": "set",
            "sequence": "ordered",
            "path": "ordered",
        }.get(reference_mode, reference_mode)
        if truth is None or reference_mode in {"none", "disabled", "region"}:
            return TaskTypeEvaluationResultV2(
                task_type="coordinate",
                metric="coordinate_region",
                score=None,
                passed=None,
                status="semantic_validator_unavailable",
                prediction_normalized=pred,
                ground_truth_normalized=None,
                details=details,
            )
        if reference_mode not in {"ordered", "set"}:
            raise ValueError(f"unsupported coordinate match mode: {reference_mode}")
        if len(pred) != len(truth):
            return TaskTypeEvaluationResultV2(
                task_type="coordinate",
                metric="coordinate_tolerance",
                score=0.0,
                passed=False,
                status="cardinality_mismatch",
                prediction_normalized=pred,
                ground_truth_normalized=truth,
                details={
                    **details,
                    "reference_cardinality": len(truth),
                    "match_mode": reference_mode,
                },
            )
        tolerance = _metadata_tolerance(
            metadata,
            "coordinate_tolerance",
            self.config.coordinate_tolerance,
        )
        errors = _coordinate_errors(pred, truth, mode=reference_mode)
        maximum_error = max(errors, default=0.0)
        passed = maximum_error <= tolerance
        return TaskTypeEvaluationResultV2(
            task_type="coordinate",
            metric="coordinate_tolerance",
            score=float(passed),
            passed=passed,
            status="evaluated",
            prediction_normalized=pred,
            ground_truth_normalized=truth,
            details={
                **details,
                "match_mode": reference_mode,
                "coordinate_tolerance": tolerance,
                "maximum_matching_error": maximum_error,
            },
        )

    def _evaluate_geometry(
        self,
        prediction: object,
        ground_truth: object,
        *,
        choices: Iterable[object] | Mapping[object, object],
        metadata: dict[str, Any],
    ) -> TaskTypeEvaluationResultV2:
        _, pred_value = _resolve_option(prediction, choices)
        _, truth_value = _resolve_option(ground_truth, choices)
        pred_relation = _canonical_relation(pred_value)
        truth_relation = _canonical_relation(truth_value)
        common_unit = metadata.get("geometry_unit", metadata.get("unit"))
        pred_vector = _parse_quantity_vector(
            pred_value,
            default_unit=metadata.get("prediction_unit", common_unit),
        )
        truth_vector = _parse_quantity_vector(
            truth_value,
            default_unit=metadata.get(
                "ground_truth_unit",
                metadata.get("reference_unit", common_unit),
            ),
        )
        if pred_relation is not None or truth_relation is not None:
            if pred_relation is None or truth_relation is None:
                return TaskTypeEvaluationResultV2(
                    task_type="geometry",
                    metric="geometry_relation",
                    score=None,
                    passed=None,
                    status="semantic_validator_unavailable",
                    prediction_normalized=pred_relation,
                    ground_truth_normalized=truth_relation,
                )
            if pred_relation != truth_relation:
                return TaskTypeEvaluationResultV2(
                    task_type="geometry",
                    metric="geometry_relation",
                    score=0.0,
                    passed=False,
                    status="evaluated",
                    prediction_normalized=pred_relation,
                    ground_truth_normalized=truth_relation,
                )
            if pred_vector is None and truth_vector is None:
                return TaskTypeEvaluationResultV2(
                    task_type="geometry",
                    metric="geometry_relation",
                    score=1.0,
                    passed=True,
                    status="evaluated",
                    prediction_normalized=pred_relation,
                    ground_truth_normalized=truth_relation,
                )
            if pred_vector is None or truth_vector is None:
                return TaskTypeEvaluationResultV2(
                    task_type="geometry",
                    metric="geometry_relation_tolerance",
                    score=None,
                    passed=None,
                    status="semantic_validator_unavailable",
                    prediction_normalized=pred_relation,
                    ground_truth_normalized=truth_relation,
                )
            metric = "geometry_relation_tolerance"
        else:
            metric = "geometry_tolerance"
        if pred_vector is None or truth_vector is None:
            return TaskTypeEvaluationResultV2(
                task_type="geometry",
                metric=metric,
                score=None,
                passed=None,
                status="semantic_validator_unavailable",
                prediction_normalized=None,
                ground_truth_normalized=None,
            )
        pred_values = tuple(item.value for item in pred_vector)
        truth_values = tuple(item.value for item in truth_vector)
        if len(pred_vector) != len(truth_vector):
            return TaskTypeEvaluationResultV2(
                task_type="geometry",
                metric=metric,
                score=0.0,
                passed=False,
                status="cardinality_mismatch",
                prediction_normalized=pred_values,
                ground_truth_normalized=truth_values,
            )
        dimensions_match = all(
            pred_item.dimension == truth_item.dimension
            for pred_item, truth_item in zip(pred_vector, truth_vector)
        )
        if not dimensions_match:
            return TaskTypeEvaluationResultV2(
                task_type="geometry",
                metric=metric,
                score=0.0,
                passed=False,
                status="unit_mismatch",
                prediction_normalized=pred_values,
                ground_truth_normalized=truth_values,
            )
        abs_tolerance = _metadata_tolerance(
            metadata,
            "geometry_tolerance",
            self.config.geometry_tolerance,
        )
        rel_tolerance = _metadata_tolerance(
            metadata,
            "geometry_rel_tolerance",
            self.config.geometry_rel_tolerance,
        )
        errors = tuple(
            abs(pred_item.value - truth_item.value)
            for pred_item, truth_item in zip(pred_vector, truth_vector)
        )
        passed = all(
            math.isclose(
                pred_item.value,
                truth_item.value,
                abs_tol=abs_tolerance,
                rel_tol=rel_tolerance,
            )
            for pred_item, truth_item in zip(pred_vector, truth_vector)
        )
        prediction_normalized: Any = pred_values
        ground_truth_normalized: Any = truth_values
        if pred_relation is not None:
            prediction_normalized = {
                "relation": pred_relation,
                "values": pred_values,
            }
            ground_truth_normalized = {
                "relation": truth_relation,
                "values": truth_values,
            }
        return TaskTypeEvaluationResultV2(
            task_type="geometry",
            metric=metric,
            score=float(passed),
            passed=passed,
            status="evaluated",
            prediction_normalized=prediction_normalized,
            ground_truth_normalized=ground_truth_normalized,
            details={
                "maximum_absolute_error": max(errors, default=0.0),
                "absolute_tolerance": abs_tolerance,
                "relative_tolerance": rel_tolerance,
            },
        )

    def _evaluate_text(
        self,
        prediction: object,
        ground_truth: object,
        *,
        choices: Iterable[object] | Mapping[object, object],
        metadata: dict[str, Any],
    ) -> TaskTypeEvaluationResultV2:
        del choices
        ignore_articles = bool(
            metadata.get("text_ignore_articles", self.config.text_ignore_articles)
        )
        pred = _semantic_text(prediction, ignore_articles=ignore_articles)
        aliases = metadata.get("semantic_aliases", metadata.get("text_aliases"))
        pred = _canonicalize_alias(pred, aliases)
        if isinstance(ground_truth, Sequence) and not isinstance(
            ground_truth,
            (str, bytes),
        ):
            truth_values = tuple(ground_truth)
        else:
            truth_values = (ground_truth,)
        accepted = metadata.get("accepted_answers")
        if isinstance(accepted, Sequence) and not isinstance(accepted, (str, bytes)):
            truth_values = (*truth_values, *accepted)
        truth_candidates = tuple(
            _canonicalize_alias(
                _semantic_text(item, ignore_articles=ignore_articles),
                aliases,
            )
            for item in truth_values
        )
        passed = bool(pred) and pred in truth_candidates
        truth_normalized: Any = (
            truth_candidates[0]
            if len(truth_candidates) == 1
            else truth_candidates
        )
        return TaskTypeEvaluationResultV2(
            task_type="text",
            metric="normalized_semantic_match",
            score=float(passed),
            passed=passed,
            status="evaluated",
            prediction_normalized=pred,
            ground_truth_normalized=truth_normalized,
            details={
                "ignore_articles": ignore_articles,
                "accepted_answer_count": len(truth_candidates),
            },
        )


# Short aliases keep configuration/result imports ergonomic without changing
# the required versioned router class name.
TaskMetricConfig = TaskMetricConfigV2
TaskTypeEvaluationResult = TaskTypeEvaluationResultV2


__all__ = [
    "ANALYSIS_ONLY",
    "CoordinateRegionValidator",
    "TASK_TYPES",
    "TaskMetricConfig",
    "TaskMetricConfigV2",
    "TaskTypeEvaluationResult",
    "TaskTypeEvaluationResultV2",
    "TaskTypeEvaluatorRouterV2",
]
