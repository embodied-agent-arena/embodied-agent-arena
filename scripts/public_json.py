"""Serialize trace data as strict JSON while preserving invalid numeric outputs."""
import json
import math


def trace_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, dict):
        return {key: trace_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [trace_value(item) for item in value]
    return value


def dumps(value):
    return json.dumps(trace_value(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
