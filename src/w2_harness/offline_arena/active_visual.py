"""Stable active-visual conditions and private media intervention plans.

The objects in this module are deliberately not wire-schema versions.  They
describe an experiment intervention before the existing offline arena builds
its normal public task and evaluator boundary.  Donor identity is available
only through explicitly private provenance; canonical/public projections carry
an opaque pairing commitment and content hashes instead.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .contracts import AssetDescriptor, PublicTask


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MEDIA_CONDITIONS = frozenset({"M0", "M1", "M2"})


class MediaInterventionError(ValueError):
    """A media condition is incomplete, mismatched, or would leak identity."""


def canonical_sha256(value: Any) -> str:
    """Hash finite JSON with the repository-wide canonical JSON convention."""

    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MediaInterventionError(
            "media intervention values must be finite JSON"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MediaInterventionError(f"{label} must be a non-empty string")
    return value.strip()


def _media_aggregate(hashes: Sequence[str]) -> str:
    return canonical_sha256({"ordered_source_media_sha256": list(hashes)})


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise MediaInterventionError("a private media source is unreadable") from exc
    return digest.hexdigest()


@dataclass(frozen=True)
class ActiveVisualArmDefinition:
    """One stable six-arm condition definition, independent of run artifacts."""

    condition_id: str
    arm_id: str
    condition_family: str
    harness_profile: str
    observation_policy: str
    intervention_kind: str
    initial_media_policy: str
    prompt_affordance: str
    submit_precondition: str | None
    diagnostic_only: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        condition_id = _nonempty(self.condition_id, "condition_id")
        arm_id = _nonempty(self.arm_id, "arm_id")
        if condition_id not in {"M0", "M1", "M2", "S0", "S1", "S2"}:
            raise MediaInterventionError("condition_id is outside the stable six arms")
        if not arm_id.startswith(f"{condition_id}_"):
            raise MediaInterventionError("arm_id must be prefixed by condition_id")
        expected_family = (
            "media_reliance"
            if condition_id.startswith("M")
            else "active_acquisition"
        )
        if self.condition_family != expected_family:
            raise MediaInterventionError("condition family does not match condition_id")
        expected_profile = "direct" if condition_id.startswith("M") else "w2_light"
        if self.harness_profile != expected_profile:
            raise MediaInterventionError(
                "harness profile does not match condition family"
            )
        for name in (
            "observation_policy",
            "intervention_kind",
            "initial_media_policy",
            "prompt_affordance",
        ):
            _nonempty(getattr(self, name), name)
        if self.submit_precondition is not None:
            _nonempty(self.submit_precondition, "submit_precondition")
        if condition_id == "S2" and self.submit_precondition is None:
            raise MediaInterventionError("S2 requires a submit precondition")
        if condition_id != "S2" and self.submit_precondition is not None:
            raise MediaInterventionError("only S2 may define a submit precondition")
        if not isinstance(self.diagnostic_only, bool):
            raise MediaInterventionError("diagnostic_only must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "arm_id": self.arm_id,
            "condition_family": self.condition_family,
            "harness_profile": self.harness_profile,
            "observation_policy": self.observation_policy,
            "intervention_kind": self.intervention_kind,
            "initial_media_policy": self.initial_media_policy,
            "prompt_affordance": self.prompt_affordance,
            "submit_precondition": self.submit_precondition,
            "diagnostic_only": self.diagnostic_only,
        }

    @property
    def canonical_hash(self) -> str:
        return canonical_sha256(self.to_dict())


SIX_ARM_DEFINITIONS = (
    ActiveVisualArmDefinition(
        condition_id="M0",
        arm_id="M0_TEXT_ONLY_DIRECT",
        condition_family="media_reliance",
        harness_profile="direct",
        observation_policy="full_context",
        intervention_kind="withhold_media",
        initial_media_policy="none",
        prompt_affordance="unchanged",
        submit_precondition=None,
    ),
    ActiveVisualArmDefinition(
        condition_id="M1",
        arm_id="M1_CORRECT_MEDIA_DIRECT",
        condition_family="media_reliance",
        harness_profile="direct",
        observation_policy="full_context",
        intervention_kind="target_media",
        initial_media_policy="all_target_media",
        prompt_affordance="unchanged",
        submit_precondition=None,
    ),
    ActiveVisualArmDefinition(
        condition_id="M2",
        arm_id="M2_SHUFFLED_MEDIA_DIRECT",
        condition_family="media_reliance",
        harness_profile="direct",
        observation_policy="full_context",
        intervention_kind="donor_media",
        initial_media_policy="all_target_slots",
        prompt_affordance="unchanged",
        submit_precondition=None,
    ),
    ActiveVisualArmDefinition(
        condition_id="S0",
        arm_id="S0_VOLUNTARY_SELECTIVE",
        condition_family="active_acquisition",
        harness_profile="w2_light",
        observation_policy="active_selective_catalog_only",
        intervention_kind="catalog_only",
        initial_media_policy="none",
        prompt_affordance="standard_optional",
        submit_precondition=None,
    ),
    ActiveVisualArmDefinition(
        condition_id="S1",
        arm_id="S1_STRENGTHENED_AFFORDANCE",
        condition_family="active_acquisition",
        harness_profile="w2_light",
        observation_policy="active_selective_catalog_only",
        intervention_kind="catalog_only",
        initial_media_policy="none",
        prompt_affordance="strengthened_optional",
        submit_precondition=None,
    ),
    ActiveVisualArmDefinition(
        condition_id="S2",
        arm_id="S2_REQUIRED_ONE_MEDIA_DIAGNOSTIC",
        condition_family="active_acquisition",
        harness_profile="w2_light",
        observation_policy="active_selective_catalog_only",
        intervention_kind="catalog_only",
        initial_media_policy="none",
        prompt_affordance="required_one_media",
        submit_precondition="one_successful_answer_bearing_media_observation",
        diagnostic_only=True,
    ),
)

ARM_DEFINITIONS = SIX_ARM_DEFINITIONS
ACTIVE_VISUAL_ARMS = MappingProxyType(
    {definition.arm_id: definition for definition in SIX_ARM_DEFINITIONS}
)
ACTIVE_VISUAL_CONDITIONS = MappingProxyType(
    {definition.condition_id: definition for definition in SIX_ARM_DEFINITIONS}
)


def get_arm_definition(value: str) -> ActiveVisualArmDefinition:
    """Resolve either a short condition ID or its stable full arm ID."""

    key = _nonempty(value, "condition or arm ID")
    definition = ACTIVE_VISUAL_CONDITIONS.get(key) or ACTIVE_VISUAL_ARMS.get(key)
    if definition is None:
        raise MediaInterventionError("unknown active-visual condition")
    return definition


@dataclass(frozen=True)
class MediaInterventionPlan:
    """A target-bound, donor-blinded request for one media intervention.

    ``to_dict`` is the canonical audit projection.  Raw donor identity and
    category are intentionally omitted and can only be obtained from the
    explicitly named :attr:`private_provenance` property.
    """

    condition_id: str
    target_benchmark_id: str
    target_sample_id: str
    donor_benchmark_id: str | None = field(default=None, repr=False)
    donor_sample_id: str | None = field(default=None, repr=False)
    donor_category: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        definition = get_arm_definition(self.condition_id)
        object.__setattr__(self, "condition_id", definition.condition_id)
        object.__setattr__(
            self,
            "target_benchmark_id",
            _nonempty(self.target_benchmark_id, "target_benchmark_id"),
        )
        object.__setattr__(
            self,
            "target_sample_id",
            _nonempty(self.target_sample_id, "target_sample_id"),
        )
        if definition.condition_id == "M2" and self.donor_benchmark_id is None:
            object.__setattr__(
                self, "donor_benchmark_id", self.target_benchmark_id
            )
        if self.donor_benchmark_id is not None:
            object.__setattr__(
                self,
                "donor_benchmark_id",
                _nonempty(self.donor_benchmark_id, "donor_benchmark_id"),
            )
        if self.donor_sample_id is not None:
            object.__setattr__(
                self,
                "donor_sample_id",
                _nonempty(self.donor_sample_id, "donor_sample_id"),
            )
        if self.donor_category is not None:
            object.__setattr__(
                self,
                "donor_category",
                _nonempty(self.donor_category, "donor_category"),
            )
        self.validate()

    @property
    def definition(self) -> ActiveVisualArmDefinition:
        return ACTIVE_VISUAL_CONDITIONS[self.condition_id]

    @property
    def arm_id(self) -> str:
        return self.definition.arm_id

    @property
    def pairing_commitment_sha256(self) -> str | None:
        if self.condition_id != "M2":
            return None
        return canonical_sha256(
            {
                "domain": "active_visual_private_donor_pairing",
                "target_benchmark_id": self.target_benchmark_id,
                "target_sample_id": self.target_sample_id,
                "donor_benchmark_id": self.donor_benchmark_id,
                "donor_sample_id": self.donor_sample_id,
                "donor_category": self.donor_category,
            }
        )

    @property
    def private_provenance(self) -> dict[str, Any]:
        """Return server-side-only assignment data; never project this to a model."""

        return {
            "intervention_plan_sha256": self.canonical_hash,
            "condition_id": self.condition_id,
            "arm_id": self.arm_id,
            "target_benchmark_id": self.target_benchmark_id,
            "target_sample_id": self.target_sample_id,
            "donor_benchmark_id": self.donor_benchmark_id,
            "donor_sample_id": self.donor_sample_id,
            "donor_category": self.donor_category,
            "donor_private_metadata_retained_server_side_only": (
                self.condition_id == "M2"
            ),
        }

    def validate(self) -> None:
        definition = self.definition
        if definition.condition_id not in _MEDIA_CONDITIONS:
            if any(
                value is not None
                for value in (
                    self.donor_benchmark_id,
                    self.donor_sample_id,
                    self.donor_category,
                )
            ):
                raise MediaInterventionError(
                    "selective conditions cannot carry a donor assignment"
                )
            return
        if self.condition_id == "M2":
            if self.donor_benchmark_id != self.target_benchmark_id:
                raise MediaInterventionError(
                    "M2 donor must come from the target benchmark"
                )
            if self.donor_sample_id is None:
                raise MediaInterventionError("M2 requires a private donor sample")
            if self.donor_sample_id == self.target_sample_id:
                raise MediaInterventionError("M2 donor must differ from target")
        elif any(
            value is not None
            for value in (
                self.donor_benchmark_id,
                self.donor_sample_id,
                self.donor_category,
            )
        ):
            raise MediaInterventionError("only M2 may carry donor assignment data")

    def validate_execution(
        self,
        *,
        harness_profile: str,
        observation_policy: str,
    ) -> None:
        if harness_profile != self.definition.harness_profile:
            raise MediaInterventionError(
                "media intervention and harness profile disagree"
            )
        if observation_policy != self.definition.observation_policy:
            raise MediaInterventionError(
                "media intervention and observation policy disagree"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical donor-blinded plan projection."""

        return {
            "condition_id": self.condition_id,
            "arm_id": self.arm_id,
            "condition_family": self.definition.condition_family,
            "intervention_kind": self.definition.intervention_kind,
            "target_benchmark_id": self.target_benchmark_id,
            "target_sample_id": self.target_sample_id,
            "pairing_commitment_sha256": self.pairing_commitment_sha256,
        }

    @property
    def canonical_hash(self) -> str:
        return canonical_sha256(self.to_dict())

    @property
    def plan_hash(self) -> str:
        return self.canonical_hash

    @property
    def hash(self) -> str:
        """Canonical hash alias used by condition-slot builders."""

        return self.canonical_hash

    def resolve(
        self,
        *,
        target_source_sample_id: str,
        target_task: PublicTask,
        target_assets: Sequence[AssetDescriptor],
        target_sources: Mapping[str, Path],
        target_visible_asset_ids: Sequence[str],
        donor_source_sample_id: str | None = None,
        donor_task: PublicTask | None = None,
        donor_assets: Sequence[AssetDescriptor] = (),
        donor_sources: Mapping[str, Path] | None = None,
        donor_visible_asset_ids: Sequence[str] = (),
    ) -> "ResolvedMediaIntervention":
        """Validate real media and produce private target-slot source bindings."""

        self.validate()
        target_source_id = _nonempty(
            target_source_sample_id, "target_source_sample_id"
        )
        if target_source_id != self.target_sample_id:
            raise MediaInterventionError("plan is bound to a different target sample")
        if target_task.benchmark_id != self.target_benchmark_id:
            raise MediaInterventionError("plan is bound to a different benchmark")

        target_by_id = _asset_map(target_task, target_assets, "target")
        target_visible = _visible_ids(
            target_visible_asset_ids, target_by_id, "target"
        )
        target_paths, target_hashes, target_modalities = _selected_media(
            target_visible, target_by_id, target_sources, "target"
        )

        donor_ids: tuple[str, ...] = ()
        donor_hashes: tuple[str, ...] = ()
        donor_modalities: tuple[str, ...] = ()
        donor_paths: tuple[Path, ...] = ()
        resolved_donor_category: str | None = None
        if self.condition_id == "M2":
            if donor_task is None or donor_sources is None:
                raise MediaInterventionError("M2 donor materialization is required")
            donor_source_id = _nonempty(
                donor_source_sample_id, "donor_source_sample_id"
            )
            if donor_source_id != self.donor_sample_id:
                raise MediaInterventionError("materialized donor differs from the plan")
            if donor_source_id == target_source_id:
                raise MediaInterventionError("M2 donor must differ from target")
            if donor_task.benchmark_id != self.target_benchmark_id:
                raise MediaInterventionError("M2 donor benchmark does not match target")
            donor_by_id = _asset_map(donor_task, donor_assets, "donor")
            donor_ids = _visible_ids(
                donor_visible_asset_ids, donor_by_id, "donor"
            )
            donor_paths, donor_hashes, donor_modalities = _selected_media(
                donor_ids, donor_by_id, donor_sources, "donor"
            )
            if len(donor_ids) != len(target_visible):
                raise MediaInterventionError(
                    "M2 donor media count does not match target"
                )
            if donor_modalities != target_modalities:
                raise MediaInterventionError("M2 donor modality does not match target")
            if _media_aggregate(donor_hashes) == _media_aggregate(target_hashes):
                raise MediaInterventionError(
                    "M2 donor media hash must differ from target"
                )
            resolved_donor_category = donor_task.category
            if (
                self.donor_category is not None
                and resolved_donor_category != self.donor_category
            ):
                raise MediaInterventionError(
                    "materialized donor category differs from private assignment"
                )
        elif any(
            value
            for value in (
                donor_source_sample_id,
                donor_task,
                donor_assets,
                donor_sources,
                donor_visible_asset_ids,
            )
        ):
            raise MediaInterventionError("non-M2 conditions cannot resolve donor media")

        if self.condition_id == "M0":
            visible_hashes: tuple[str, ...] = ()
            source_bindings: tuple[tuple[str, Path], ...] = ()
            source_hash_overrides: tuple[tuple[str, str], ...] = ()
            initial_visible: tuple[str, ...] = ()
        elif self.condition_id == "M1":
            visible_hashes = target_hashes
            source_bindings = tuple(zip(target_visible, target_paths))
            source_hash_overrides = ()
            initial_visible = target_visible
        else:
            visible_hashes = donor_hashes
            source_bindings = tuple(zip(target_visible, donor_paths))
            source_hash_overrides = tuple(zip(target_visible, donor_hashes))
            initial_visible = target_visible

        slots = tuple(
            MediaInterventionSlot(
                target_asset_id=asset_id,
                modality=target_modalities[index],
                target_source_media_hash=target_hashes[index],
                visible_source_media_hash=(
                    None
                    if self.condition_id == "M0"
                    else visible_hashes[index]
                ),
            )
            for index, asset_id in enumerate(target_visible)
        )
        private_provenance = {
            **self.private_provenance,
            "target_category": target_task.category,
            "target_asset_ids": list(target_visible),
            "target_media_sha256": list(target_hashes),
            "target_media_aggregate_sha256": _media_aggregate(target_hashes),
            "donor_sample_id": donor_source_sample_id,
            "donor_category": resolved_donor_category,
            "donor_asset_ids": list(donor_ids),
            "donor_media_sha256": list(donor_hashes),
            "donor_media_aggregate_sha256": (
                None if not donor_hashes else _media_aggregate(donor_hashes)
            ),
            "target_to_donor_asset_ids": (
                []
                if self.condition_id != "M2"
                else [
                    {
                        "target_asset_id": target_id,
                        "donor_asset_id": donor_id,
                    }
                    for target_id, donor_id in zip(target_visible, donor_ids)
                ]
            ),
        }
        return ResolvedMediaIntervention(
            plan=self,
            slots=slots,
            initial_visible_asset_ids=initial_visible,
            _source_bindings=source_bindings,
            _source_hash_overrides=source_hash_overrides,
            _private_provenance=private_provenance,
        )


@dataclass(frozen=True)
class MediaInterventionSlot:
    """Donor-blinded content binding for one target public asset slot."""

    target_asset_id: str
    modality: str
    target_source_media_hash: str
    visible_source_media_hash: str | None

    def __post_init__(self) -> None:
        _nonempty(self.target_asset_id, "target_asset_id")
        _nonempty(self.modality, "modality")
        if _SHA256.fullmatch(self.target_source_media_hash) is None:
            raise MediaInterventionError("target source media hash is invalid")
        if (
            self.visible_source_media_hash is not None
            and _SHA256.fullmatch(self.visible_source_media_hash) is None
        ):
            raise MediaInterventionError("visible source media hash is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_asset_id": self.target_asset_id,
            "modality": self.modality,
            "target_source_media_hash": self.target_source_media_hash,
            "visible_source_media_hash": self.visible_source_media_hash,
        }


@dataclass(frozen=True)
class ResolvedMediaIntervention:
    """Validated runtime bindings; paths and donor provenance remain private."""

    plan: MediaInterventionPlan
    slots: tuple[MediaInterventionSlot, ...]
    initial_visible_asset_ids: tuple[str, ...]
    _source_bindings: tuple[tuple[str, Path], ...] = field(repr=False, compare=False)
    _source_hash_overrides: tuple[tuple[str, str], ...] = field(
        repr=False, compare=False
    )
    _private_provenance: Mapping[str, Any] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        self.validate()

    @property
    def source_bindings(self) -> dict[str, Path]:
        return dict(self._source_bindings)

    @property
    def source_hash_overrides(self) -> dict[str, str]:
        return dict(self._source_hash_overrides)

    @property
    def source_content_sha256_overrides(self) -> dict[str, str]:
        return dict(self._source_hash_overrides)

    @property
    def private_provenance(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._private_provenance, sort_keys=True))

    def validate(self) -> None:
        target_ids = tuple(slot.target_asset_id for slot in self.slots)
        if not target_ids or len(set(target_ids)) != len(target_ids):
            raise MediaInterventionError(
                "resolved target slots must be non-empty and unique"
            )
        if not set(self.initial_visible_asset_ids) <= set(target_ids):
            raise MediaInterventionError(
                "initial media contains an unknown target slot"
            )
        binding_ids = tuple(asset_id for asset_id, _ in self._source_bindings)
        override_ids = tuple(asset_id for asset_id, _ in self._source_hash_overrides)
        if self.plan.condition_id == "M0":
            if self.initial_visible_asset_ids or binding_ids or override_ids:
                raise MediaInterventionError("M0 must expose zero initial media")
        elif self.plan.condition_id == "M1":
            if binding_ids != target_ids or override_ids:
                raise MediaInterventionError("M1 must bind every target slot exactly")
            if any(
                slot.visible_source_media_hash != slot.target_source_media_hash
                for slot in self.slots
            ):
                raise MediaInterventionError("M1 visible media must equal target media")
        elif self.plan.condition_id == "M2":
            if binding_ids != target_ids or override_ids != target_ids:
                raise MediaInterventionError("M2 must privately bind every target slot")
            if _media_aggregate(
                tuple(slot.target_source_media_hash for slot in self.slots)
            ) == _media_aggregate(
                tuple(str(slot.visible_source_media_hash) for slot in self.slots)
            ):
                raise MediaInterventionError("M2 aggregate media hash must differ")
        else:  # pragma: no cover - resolve rejects selective conditions first.
            raise MediaInterventionError("resolved intervention must be M0, M1, or M2")

    def to_dict(self) -> dict[str, Any]:
        target_hashes = tuple(slot.target_source_media_hash for slot in self.slots)
        visible_hashes = tuple(
            slot.visible_source_media_hash
            for slot in self.slots
            if slot.visible_source_media_hash is not None
        )
        return {
            **self.plan.to_dict(),
            "target_asset_ids": [slot.target_asset_id for slot in self.slots],
            "target_media_count": len(self.slots),
            "visible_media_count": len(visible_hashes),
            "initial_visible_asset_ids": list(self.initial_visible_asset_ids),
            "target_media_aggregate_sha256": _media_aggregate(target_hashes),
            "visible_media_aggregate_sha256": (
                None if not visible_hashes else _media_aggregate(visible_hashes)
            ),
            "slots": [slot.to_dict() for slot in self.slots],
        }

    @property
    def canonical_hash(self) -> str:
        return canonical_sha256(self.to_dict())


def _asset_map(
    task: PublicTask,
    assets: Sequence[AssetDescriptor],
    label: str,
) -> dict[str, AssetDescriptor]:
    values = tuple(assets)
    by_id = {asset.asset_id: asset for asset in values}
    if len(by_id) != len(values) or set(task.asset_ids) != set(by_id):
        raise MediaInterventionError(f"{label} task and catalog disagree")
    return by_id


def _visible_ids(
    values: Sequence[str],
    assets: Mapping[str, AssetDescriptor],
    label: str,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise MediaInterventionError(f"{label} visible assets must be a sequence")
    result = tuple(str(value) for value in values)
    if not result or len(set(result)) != len(result) or not set(result) <= set(assets):
        raise MediaInterventionError(
            f"{label} visible asset order is empty, duplicated, or unknown"
        )
    return result


def _selected_media(
    asset_ids: Sequence[str],
    assets: Mapping[str, AssetDescriptor],
    sources: Mapping[str, Path],
    label: str,
) -> tuple[tuple[Path, ...], tuple[str, ...], tuple[str, ...]]:
    paths: list[Path] = []
    hashes: list[str] = []
    modalities: list[str] = []
    for asset_id in asset_ids:
        descriptor = assets[asset_id]
        try:
            path = Path(sources[asset_id]).resolve()
        except (KeyError, TypeError, ValueError) as exc:
            raise MediaInterventionError(
                f"{label} media source binding is missing"
            ) from exc
        if not path.is_file():
            raise MediaInterventionError(f"{label} media source is unreadable")
        if _sha256_file(path) != descriptor.source_media_hash:
            raise MediaInterventionError(
                f"{label} media source hash does not match its descriptor"
            )
        paths.append(path)
        hashes.append(descriptor.source_media_hash)
        modalities.append(descriptor.modality)
    return tuple(paths), tuple(hashes), tuple(modalities)


__all__ = [
    "ACTIVE_VISUAL_ARMS",
    "ACTIVE_VISUAL_CONDITIONS",
    "ARM_DEFINITIONS",
    "ActiveVisualArmDefinition",
    "MediaInterventionError",
    "MediaInterventionPlan",
    "MediaInterventionSlot",
    "ResolvedMediaIntervention",
    "SIX_ARM_DEFINITIONS",
    "canonical_sha256",
    "get_arm_definition",
]
