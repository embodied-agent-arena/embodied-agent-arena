"""Canonical inventory of benchmark directory targets.

The catalog deliberately models a directory target separately from a runtime
target.  Most directories expose one runtime id, while the ESI directory owns
both the ``esi_spatial`` fixture target and the ``vsi_bench`` dataset target.
Keeping this inventory independent of runtime imports also makes it safe to use
from packaging, CI, and lightweight OpenHands setup code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from .paths import get_project_paths

BenchmarkKind = Literal["interactive_benchmark", "diagnostic_benchmark", "policy_skill"]
_VALID_KINDS = frozenset({"interactive_benchmark", "diagnostic_benchmark", "policy_skill"})
DEFAULT_BENCHMARK_ROOT = get_project_paths().benchmark_root


class BenchmarkCatalogError(ValueError):
    """Raised when the checked-in catalog and benchmark directories diverge."""


@dataclass(frozen=True, slots=True)
class BenchmarkCatalogEntry:
    """One checked-in benchmark directory and the runtime targets it owns."""

    benchmark_id: str
    family: str
    scope: str
    path: str
    kind: BenchmarkKind
    official_success_semantics: str
    runtime_target_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["runtime_target_ids"] = list(self.runtime_target_ids)
        return payload


BENCHMARK_CATALOG: tuple[BenchmarkCatalogEntry, ...] = (
    BenchmarkCatalogEntry(
        "mmsi_bench", "MMSI-Bench", "non_operation", "benchmarks/non_operation/mmsi_bench",
        "diagnostic_benchmark", "Exact submitted answer match against the harness-side spatial answer key.",
        ("mmsi_bench",),
    ),
    BenchmarkCatalogEntry(
        "alfworld", "ALFWorld", "non_operation", "benchmarks/non_operation/alfworld",
        "interactive_benchmark", "Harness-side TextWorld state reports won=True and done=True.", ("alfworld",),
    ),
    BenchmarkCatalogEntry(
        "scienceworld", "ScienceWorld", "non_operation", "benchmarks/non_operation/scienceworld",
        "interactive_benchmark", "Harness-side environment reports done=True with score >= 100.", ("scienceworld",),
    ),
    BenchmarkCatalogEntry(
        "maniskill", "ManiSkill3", "operation", "benchmarks/operation/maniskill",
        "interactive_benchmark", "The harness-only ManiSkill task verifier reports success after agent actions.", ("maniskill",),
    ),
    BenchmarkCatalogEntry(
        "vimabench", "VIMA-Bench", "operation", "benchmarks/operation/vimabench",
        "interactive_benchmark", "The harness consumes the native environment task reward/success signal.", ("vimabench",),
    ),
    BenchmarkCatalogEntry(
        "cliport", "CLIPort", "operation", "benchmarks/operation/cliport",
        "interactive_benchmark", "The harness consumes reward and done returned by the native env.step call.", ("cliport",),
    ),
    BenchmarkCatalogEntry(
        "vlabench", "VLABench", "operation", "benchmarks/operation/vlabench",
        "interactive_benchmark", "Harness-side termination and evaluator success_rate/intention_score/progress_score determine success.",
        ("vlabench",),
    ),
    BenchmarkCatalogEntry(
        "robocasa", "RoboCasa", "operation", "benchmarks/operation/robocasa",
        "interactive_benchmark", "Harness-side info['success'] or the private official task predicate reports success.", ("robocasa",),
    ),
    BenchmarkCatalogEntry(
        "capx", "CaP-X", "operation", "benchmarks/operation/capx",
        "interactive_benchmark", "The outer harness reads native task reward and completion after the rollout.", ("capx",),
    ),
    BenchmarkCatalogEntry(
        "rlbench", "RLBench", "operation", "benchmarks/operation/rlbench",
        "interactive_benchmark", "The harness consumes the native task success condition or reward/termination state.", ("rlbench",),
    ),
    BenchmarkCatalogEntry(
        "calvin", "CALVIN", "operation", "benchmarks/operation/calvin",
        "interactive_benchmark", "Harness-side official chained subgoal/task predicates determine rollout success.", ("calvin",),
    ),
    BenchmarkCatalogEntry(
        "openvla", "OpenVLA", "policy_contract", "benchmarks/policy_contract/openvla",
        "policy_skill", "No standalone task success: downstream benchmark official verification scores the policy rollout.", ("openvla",),
    ),
    BenchmarkCatalogEntry(
        "openpi", "OpenPI", "policy_contract", "benchmarks/policy_contract/openpi",
        "policy_skill", "No standalone task success: downstream benchmark official verification scores the policy rollout.", ("openpi",),
    ),
    BenchmarkCatalogEntry(
        "lerobot", "LeRobot", "policy_contract", "benchmarks/policy_contract/lerobot",
        "policy_skill", "No standalone task success: downstream benchmark official verification scores the policy rollout.", ("lerobot",),
    ),
    BenchmarkCatalogEntry(
        "octo", "Octo", "policy_contract", "benchmarks/policy_contract/octo",
        "policy_skill", "No standalone task success: downstream benchmark official verification scores the policy rollout.", ("octo",),
    ),
    BenchmarkCatalogEntry(
        "behavior1k", "BEHAVIOR-1K", "operation", "benchmarks/operation/behavior1k",
        "interactive_benchmark", "The harness reads official OmniGibson/BDDL env.task.success after execution.", ("behavior1k",),
    ),
    BenchmarkCatalogEntry(
        "esi", "ESI / VSI-Bench", "non_operation", "benchmarks/non_operation/esi",
        "diagnostic_benchmark", "The submitted answer must match the harness-side spatial oracle.",
        ("esi_spatial", "vsi_bench"),
    ),
    BenchmarkCatalogEntry(
        "robocasa365", "RoboCasa365", "operation", "benchmarks/operation/robocasa365",
        "interactive_benchmark", "The harness-only official task evaluator reports success after agent actions.", ("robocasa365",),
    ),
    BenchmarkCatalogEntry(
        "spatialclaw", "SpatialClaw", "non_operation", "benchmarks/non_operation/spatialclaw",
        "diagnostic_benchmark", "The submitted answer must match the harness-side spatial answer key.", ("spatialclaw",),
    ),
    BenchmarkCatalogEntry(
        "robowits", "RoboWits", "operation", "benchmarks/operation/robowits",
        "interactive_benchmark", "The harness-only official task verifier must return success after execution.", ("robowits",),
    ),
    BenchmarkCatalogEntry(
        "robotwin2", "RoboTwin2", "operation", "benchmarks/operation/robotwin2",
        "interactive_benchmark", "The private official check_success predicate returns success after submitted actions.", ("robotwin2",),
    ),
    BenchmarkCatalogEntry(
        "robodojo", "RoboDojo", "operation", "benchmarks/operation/robodojo",
        "interactive_benchmark", "The harness-only official verifier reports success after action execution.", ("robodojo",),
    ),
)


def discover_benchmark_directories(benchmark_root: Path | str = DEFAULT_BENCHMARK_ROOT) -> tuple[str, ...]:
    """Return repository-relative directory targets found directly below waves.

    Private/shared directories (names beginning with ``_``) are infrastructure,
    not benchmark targets.  Every other directory directly below a wave is a
    target and therefore must be registered.
    """

    root = Path(benchmark_root)
    if not root.is_dir():
        raise BenchmarkCatalogError(f"benchmark root does not exist or is not a directory: {root}")
    targets = (
        f"benchmarks/{scope.name}/{target.name}"
        for scope in root.iterdir()
        if scope.is_dir() and scope.name in {"operation", "non_operation", "policy_contract"}
        for target in scope.iterdir()
        if target.is_dir() and not target.name.startswith("_")
    )
    return tuple(sorted(targets))


def validate_benchmark_catalog(benchmark_root: Path | str = DEFAULT_BENCHMARK_ROOT) -> tuple[BenchmarkCatalogEntry, ...]:
    """Strictly validate schema, uniqueness, paths, and filesystem coverage."""

    errors: list[str] = []
    ids = [entry.benchmark_id for entry in BENCHMARK_CATALOG]
    paths = [entry.path for entry in BENCHMARK_CATALOG]
    runtime_ids = [runtime_id for entry in BENCHMARK_CATALOG for runtime_id in entry.runtime_target_ids]
    for label, values in (("benchmark_id", ids), ("path", paths), ("runtime_target_id", runtime_ids)):
        duplicates = sorted({value for value in values if values.count(value) > 1})
        if duplicates:
            errors.append(f"duplicate {label}(s): {duplicates}")

    for entry in BENCHMARK_CATALOG:
        parts = Path(entry.path).parts
        if len(parts) != 3 or parts[0] != "benchmarks":
            errors.append(f"{entry.benchmark_id}: path must be benchmarks/<wave>/<target>: {entry.path}")
        elif parts[1] != entry.scope:
            errors.append(f"{entry.benchmark_id}: scope {entry.scope!r} does not match path {entry.path!r}")
        if entry.kind not in _VALID_KINDS:
            errors.append(f"{entry.benchmark_id}: invalid kind {entry.kind!r}")
        if not entry.family.strip() or not entry.official_success_semantics.strip():
            errors.append(f"{entry.benchmark_id}: family and official_success_semantics must be non-empty")
        if not entry.runtime_target_ids or any(not item.strip() for item in entry.runtime_target_ids):
            errors.append(f"{entry.benchmark_id}: runtime_target_ids must contain non-empty ids")

    discovered = set(discover_benchmark_directories(benchmark_root))
    registered = set(paths)
    unregistered = sorted(discovered - registered)
    missing = sorted(registered - discovered)
    if unregistered:
        errors.append(f"unregistered benchmark directories: {unregistered}")
    if missing:
        errors.append(f"registered benchmark directories missing from disk: {missing}")
    if errors:
        raise BenchmarkCatalogError("benchmark catalog validation failed: " + "; ".join(errors))
    return BENCHMARK_CATALOG


def benchmark_catalog_manifest(benchmark_root: Path | str = DEFAULT_BENCHMARK_ROOT) -> dict[str, object]:
    """Build a JSON-serializable, validation-gated catalog manifest."""

    entries = validate_benchmark_catalog(benchmark_root)
    return {
        "schema_version": "embodied-harness-benchmark-catalog/v1",
        "directory_target_count": len(entries),
        "runtime_target_count": sum(len(entry.runtime_target_ids) for entry in entries),
        "benchmarks": [entry.to_dict() for entry in entries],
    }


__all__ = [
    "BENCHMARK_CATALOG",
    "DEFAULT_BENCHMARK_ROOT",
    "BenchmarkCatalogEntry",
    "BenchmarkCatalogError",
    "BenchmarkKind",
    "benchmark_catalog_manifest",
    "discover_benchmark_directories",
    "validate_benchmark_catalog",
]
