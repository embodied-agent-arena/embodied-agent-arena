"""Small, dependency-free bridge for per-task 1% pool coordinates.

The lightweight manifest deliberately keeps the benchmark adapter ``case_id``
stable and puts the selected upstream coordinate in worker environment
variables.  This module is the narrow seam between those two layers.  It does
not import a simulator and it does not assume that an adapter can consume an
arbitrary task name.  Callers can therefore attach provenance to a
``TaskSpec`` for every task while opting in to a real adapter binding only when
the adapter implements ``bind_pool_coordinate``.

The bridge is intentionally conservative:

* no pool variables means ``read_pool_coordinate`` returns ``None`` and the
  pre-pool representative behavior is unchanged;
* malformed integer coordinates fail closed with a useful ``ValueError``;
* no unknown keys are inserted into a benchmark's reset config (most runtime
  dataclasses and Gym constructors reject them);
* an adapter may implement an optional ``bind_pool_coordinate(payload)`` hook
  when it has a genuine upstream task/episode mapping.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from typing import Any, Mapping


JsonDict = dict[str, Any]

POOL_SCHEMA_ENV = "EMBODIED_ARENA_POOL_SCHEMA"
POOL_ENTRY_ID_ENV = "EMBODIED_ARENA_POOL_ENTRY_ID"
POOL_TASK_ID_ENV = "EMBODIED_ARENA_POOL_TASK_ID"
POOL_VARIATION_ENV = "EMBODIED_ARENA_POOL_VARIATION"
POOL_SEED_ENV = "EMBODIED_ARENA_POOL_SEED"
POOL_STRATUM_ENV = "EMBODIED_ARENA_POOL_STRATUM"
POOL_SELECTION_RANK_ENV = "EMBODIED_ARENA_POOL_SELECTION_RANK"
POOL_RANK_KEY_ENV = "EMBODIED_ARENA_POOL_RANK_KEY"
POOL_DENOMINATOR_ENV = "EMBODIED_ARENA_POOL_DENOMINATOR"
POOL_TARGET_CASES_ENV = "EMBODIED_ARENA_POOL_TARGET_CASES"
POOL_SOURCE_REVISION_ENV = "EMBODIED_ARENA_POOL_SOURCE_REVISION"
POOL_SAMPLING_SEED_ENV = "EMBODIED_ARENA_POOL_SAMPLING_SEED"

# These are set by ``full_evaluation.execute_task`` for every task, including
# non-pool tasks.  They are fallbacks only after a pool marker is present.
EVALUATION_VARIATION_ENV = "EMBODIED_ARENA_VARIATION"
EVALUATION_SEED_ENV = "EMBODIED_ARENA_EVALUATION_SEED"


@dataclass(frozen=True, slots=True)
class PoolCoordinate:
    """One immutable coordinate selected from a benchmark's frozen pool."""

    schema: str | None = None
    entry_id: str | None = None
    task_id: str | None = None
    variation: str | None = None
    seed: int | None = None
    stratum: str | None = None
    selection_rank: int | None = None
    rank_key: str | None = None
    denominator: int | None = None
    target_cases: int | None = None
    source_revision: str | None = None
    sampling_seed: int | None = None

    @property
    def active(self) -> bool:
        """Whether this represents a marked pool task rather than no context."""

        return bool(self.schema or self.entry_id or self.task_id)

    def to_dict(self) -> JsonDict:
        """Return a JSON-safe payload, omitting absent optional dimensions."""

        payload = {
            key: value for key, value in asdict(self).items() if value is not None
        }
        # The manifest calls this field ``schema_version``; retain the short
        # ``schema`` attribute for the Python API while emitting the canonical
        # wire name as an alias for downstream aggregators.
        if self.schema is not None:
            payload["schema_version"] = self.schema
        return payload


class PoolCoordinateError(ValueError):
    """Raised when a marked pool coordinate cannot be parsed safely."""


def _text(environment: Mapping[str, Any], key: str) -> str | None:
    value = environment.get(key)
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _integer(
    environment: Mapping[str, Any],
    key: str,
    *,
    default: int | None = None,
) -> int | None:
    value = _text(environment, key)
    if value is None:
        return default
    try:
        # bool is not representable through an environment string, and
        # accepting surrounding whitespace is useful for hand-launched jobs.
        return int(value, 10)
    except (TypeError, ValueError) as exc:
        raise PoolCoordinateError(
            f"{key} must be an integer for a marked pool task; got {value!r}"
        ) from exc


def read_pool_coordinate(
    environment: Mapping[str, Any] | None = None,
    *,
    fallback_seed: int | None = None,
) -> PoolCoordinate | None:
    """Read a pool coordinate from ``environment``.

    ``None`` is returned for ordinary representative runs.  A pool marker is
    present when any of schema, entry id, or pool task id is set.  The
    generated manifest emits explicit pool variation and seed values; the
    scheduler's evaluation variation/seed names remain compatibility
    fallbacks for hand-launched workers.
    """

    source: Mapping[str, Any] = os.environ if environment is None else environment
    schema = _text(source, POOL_SCHEMA_ENV)
    entry_id = _text(source, POOL_ENTRY_ID_ENV)
    task_id = _text(source, POOL_TASK_ID_ENV)
    if not (schema or entry_id or task_id):
        return None

    variation = _text(source, POOL_VARIATION_ENV) or _text(
        source, EVALUATION_VARIATION_ENV
    )
    seed = _integer(source, POOL_SEED_ENV)
    if seed is None:
        # A marked pool task launched by full_evaluation always gets this
        # variable.  If a direct caller omits it, retain the catalog seed.
        seed = _integer(source, EVALUATION_SEED_ENV, default=fallback_seed)

    return PoolCoordinate(
        schema=schema,
        entry_id=entry_id,
        task_id=task_id,
        variation=variation,
        seed=seed,
        stratum=_text(source, POOL_STRATUM_ENV),
        selection_rank=_integer(source, POOL_SELECTION_RANK_ENV),
        rank_key=_text(source, POOL_RANK_KEY_ENV),
        denominator=_integer(source, POOL_DENOMINATOR_ENV),
        target_cases=_integer(source, POOL_TARGET_CASES_ENV),
        source_revision=_text(source, POOL_SOURCE_REVISION_ENV),
        sampling_seed=_integer(source, POOL_SAMPLING_SEED_ENV),
    )


def effective_seed(
    catalog_seed: int | None,
    explicit_seed: int | None,
    coordinate: PoolCoordinate | None,
) -> int | None:
    """Choose the reset seed, preferring the selected pool coordinate."""

    if coordinate is not None and coordinate.seed is not None:
        return coordinate.seed
    if explicit_seed is not None:
        return explicit_seed
    return catalog_seed


def bind_pool_coordinate(
    backend: Any,
    coordinate: PoolCoordinate | None,
) -> JsonDict:
    """Invoke an optional adapter binding hook without requiring one.

    Existing adapters do not implement the hook and therefore receive a
    ``coordinate_only`` status.  A future adapter can implement
    ``bind_pool_coordinate(dict)`` and return a JSON-safe status payload; the
    bridge wraps non-dict returns so the caller can always record the result.
    Exceptions are intentionally converted to a status object so the native
    loop can include the exact blocker in its normal attempt report.
    """

    if coordinate is None:
        return {"mode": "none", "bound": False}
    hook = getattr(backend, "bind_pool_coordinate", None)
    if not callable(hook):
        return {
            "mode": "coordinate_only",
            "bound": False,
            "reason": "adapter_has_no_bind_pool_coordinate_hook",
        }
    try:
        result = hook(coordinate.to_dict())
    except Exception as exc:  # noqa: BLE001 - preserve adapter blocker in report
        return {
            "mode": "adapter_hook",
            "bound": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    payload: JsonDict = (
        dict(result) if isinstance(result, Mapping) else {"result": result}
    )
    payload.setdefault("mode", "adapter_hook")
    payload.setdefault("bound", True)
    return payload


def attach_pool_coordinate(
    task: Any,
    coordinate: PoolCoordinate | None,
    *,
    case_id: str | None = None,
    canonical_task_id: str | None = None,
    canonical_seed: int | None = None,
    reset_config: Mapping[str, Any] | None = None,
    binding: Mapping[str, Any] | None = None,
) -> Any:
    """Attach pool provenance to a mutable ``TaskSpec`` and return it.

    The payload is deliberately metadata-only: injecting an unknown key into
    an adapter dataclass or Gym constructor would make old representative
    cases fail.  ``pool_reset_context`` records which reset config was actually
    used and leaves the canonical reset config untouched.
    """

    if coordinate is None:
        return task
    metadata = getattr(task, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
        setattr(task, "metadata", metadata)
    binding_payload = dict(binding or {"mode": "coordinate_only", "bound": False})
    metadata["pool_coordinate"] = coordinate.to_dict()
    metadata["pool_reset_context"] = {
        "mode": str(binding_payload.get("mode") or "coordinate_only"),
        "adapter_case_id": case_id,
        "canonical_task_id": canonical_task_id,
        "canonical_seed": canonical_seed,
        "reset_config_keys": sorted(str(key) for key in (reset_config or {})),
        "binding": binding_payload,
    }
    return task


def record_pool_binding(
    backend: Any,
    coordinate: PoolCoordinate | None,
    binding: Mapping[str, Any] | None = None,
) -> None:
    """Append one non-agent-visible trace event when a pool is active."""

    if coordinate is None:
        return
    recorder = getattr(backend, "record_event", None)
    if not callable(recorder):
        return
    recorder(
        "pool_coordinate_binding",
        {
            "coordinate": coordinate.to_dict(),
            "binding": dict(binding or {}),
        },
    )


__all__ = [
    "EVALUATION_SEED_ENV",
    "EVALUATION_VARIATION_ENV",
    "POOL_COORDINATE_ENV_KEYS",
    "POOL_DENOMINATOR_ENV",
    "POOL_ENTRY_ID_ENV",
    "POOL_RANK_KEY_ENV",
    "POOL_SAMPLING_SEED_ENV",
    "POOL_SCHEMA_ENV",
    "POOL_SEED_ENV",
    "POOL_SELECTION_RANK_ENV",
    "POOL_SOURCE_REVISION_ENV",
    "POOL_STRATUM_ENV",
    "POOL_TARGET_CASES_ENV",
    "POOL_TASK_ID_ENV",
    "POOL_VARIATION_ENV",
    "PoolCoordinate",
    "PoolCoordinateError",
    "attach_pool_coordinate",
    "bind_pool_coordinate",
    "effective_seed",
    "read_pool_coordinate",
    "record_pool_binding",
]


POOL_COORDINATE_ENV_KEYS = (
    POOL_SCHEMA_ENV,
    POOL_ENTRY_ID_ENV,
    POOL_TASK_ID_ENV,
    POOL_VARIATION_ENV,
    POOL_SEED_ENV,
    POOL_STRATUM_ENV,
    POOL_SELECTION_RANK_ENV,
    POOL_RANK_KEY_ENV,
    POOL_DENOMINATOR_ENV,
    POOL_TARGET_CASES_ENV,
    POOL_SOURCE_REVISION_ENV,
    POOL_SAMPLING_SEED_ENV,
)
