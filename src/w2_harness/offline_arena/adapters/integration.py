"""Typed bridge from source adapters into the single offline environment."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..active_visual import (
    MediaInterventionError,
    MediaInterventionPlan,
    ResolvedMediaIntervention,
)
from ..contracts import (
    ActionPolicy,
    ActionType,
    AssetDescriptor,
    EvaluationResult,
    MODALITY_SUFFICIENT_EVIDENCE,
    OfflineArenaContractError,
    ParsedSubmission,
    PublicTask,
)
from ..environment import OfflineEvidenceEnvironment
from ..evaluator import PrivateEvaluationError, ReferenceAccessError
from ..media_transport import MediaTransport, MediaTransportRecord
from ..state import EpisodeLifecycleState
from ..visual_sufficiency import (
    EvidenceSufficiencyRequirement,
    ModalityEvidenceBundle,
)
from .base import AdapterSample, BaseAdapter


_ACTIONS_BY_BENCHMARK: dict[str, tuple[ActionType, ...]] = {
    "MMSI-Bench": (
        ActionType.LIST_ASSETS,
        ActionType.GET_VIEW,
        ActionType.COMPOSE_ASSETS,
        ActionType.CROP_REGION,
        ActionType.RECORD_EVIDENCE,
        ActionType.SUBMIT,
    ),
    "VSI-Bench": (
        ActionType.LIST_ASSETS,
        ActionType.GET_FRAME,
        ActionType.GET_FRAME_WINDOW,
        ActionType.CROP_REGION,
        ActionType.RECORD_EVIDENCE,
        ActionType.SUBMIT,
    ),
    "MindCube": (
        ActionType.LIST_ASSETS,
        ActionType.GET_VIEW,
        ActionType.COMPOSE_ASSETS,
        ActionType.RECORD_EVIDENCE,
        ActionType.QUERY_STATE,
        ActionType.SUBMIT,
    ),
    "3DSRBench": (
        ActionType.OPEN_ASSET,
        ActionType.CROP_REGION,
        ActionType.ZOOM_REGION,
        ActionType.RECORD_EVIDENCE,
        ActionType.SUBMIT,
    ),
    "RoboSpatial-Home": (
        ActionType.OPEN_ASSET,
        ActionType.CROP_REGION,
        ActionType.QUERY_STATE,
        ActionType.RECORD_EVIDENCE,
        ActionType.SUBMIT,
    ),
    "BOP-ASK": (
        ActionType.OPEN_ASSET,
        ActionType.CROP_REGION,
        ActionType.QUERY_STATE,
        ActionType.RECORD_EVIDENCE,
        ActionType.SUBMIT,
    ),
}

_ACTIVE_SELECTIVE_ACTIONS_BY_BENCHMARK: dict[str, tuple[ActionType, ...]] = {
    "MMSI-Bench": (
        ActionType.LIST_ASSETS,
        ActionType.GET_VIEW,
        ActionType.COMPOSE_ASSETS,
        ActionType.CROP_REGION,
        ActionType.QUERY_STATE,
        ActionType.SUBMIT,
    ),
    "MindCube": (
        ActionType.LIST_ASSETS,
        ActionType.GET_VIEW,
        ActionType.COMPOSE_ASSETS,
        ActionType.CROP_REGION,
        ActionType.QUERY_STATE,
        ActionType.SUBMIT,
    ),
    "VSI-Bench": (
        ActionType.LIST_ASSETS,
        ActionType.GET_FRAME,
        ActionType.GET_FRAME_WINDOW,
        ActionType.CROP_REGION,
        ActionType.QUERY_STATE,
        ActionType.SUBMIT,
    ),
    "3DSRBench": (
        ActionType.LIST_ASSETS,
        ActionType.OPEN_ASSET,
        ActionType.CROP_REGION,
        ActionType.ZOOM_REGION,
        ActionType.QUERY_STATE,
        ActionType.SUBMIT,
    ),
    "RoboSpatial-Home": (
        ActionType.LIST_ASSETS,
        ActionType.OPEN_ASSET,
        ActionType.CROP_REGION,
        ActionType.ZOOM_REGION,
        ActionType.QUERY_STATE,
        ActionType.SUBMIT,
    ),
    "BOP-ASK": (
        ActionType.LIST_ASSETS,
        ActionType.OPEN_ASSET,
        ActionType.CROP_REGION,
        ActionType.ZOOM_REGION,
        ActionType.QUERY_STATE,
        ActionType.SUBMIT,
    ),
}


def _sufficiency_actions(
    requirement: EvidenceSufficiencyRequirement,
) -> tuple[ActionType, ...]:
    """Bind the existing public action surface to the typed modality contract."""

    media_actions = tuple(ActionType(item) for item in requirement.allowed_actions)
    return tuple(
        dict.fromkeys(
            (
                ActionType.LIST_ASSETS,
                *media_actions,
                ActionType.QUERY_STATE,
                ActionType.SUBMIT,
            )
        )
    )


def _implementation_hash(adapter: BaseAdapter) -> str:
    identity = (
        f"{type(adapter).__module__}:{type(adapter).__qualname__}:"
        "w2-adapter-private-evaluator-gateway-v1.0"
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


class AdapterPrivateEvaluatorGateway:
    """Open adapter-owned truth only after the environment reaches SUBMITTED."""

    def __init__(self, adapter: BaseAdapter, sample: AdapterSample) -> None:
        self._adapter = adapter
        self._sample = sample
        self._discarded = False
        self._evaluated = False
        self._reference_access_count = 0

    @property
    def reference_access_count(self) -> int:
        return self._reference_access_count

    def evaluate(
        self,
        submission: ParsedSubmission,
        *,
        state: EpisodeLifecycleState,
        task: PublicTask,
        episode_id: str,
    ) -> EvaluationResult:
        if state is not EpisodeLifecycleState.SUBMITTED:
            raise ReferenceAccessError(
                "adapter private evaluation requires SUBMITTED state"
            )
        if self._discarded or self._evaluated:
            raise PrivateEvaluationError("adapter private evaluator is no longer active")
        self._reference_access_count += 1
        try:
            submission_payload: dict[str, Any] = {
                "answer": submission.answer,
                "evidence_refs": list(submission.evidence_refs),
            }
            if submission.confidence is not None:
                submission_payload["confidence"] = submission.confidence
            parsed = self._adapter.parse_submission(
                self._sample,
                submission_payload,
            )
            raw = self._adapter.evaluate_private(self._sample, parsed)
        except Exception:
            raise PrivateEvaluationError("adapter private evaluation failed") from None
        if not isinstance(raw, Mapping):
            raise PrivateEvaluationError("adapter evaluator returned a non-object")

        submission_valid = raw.get("submission_valid") is True
        evaluated = raw.get("evaluated", submission_valid) is True
        correct = raw.get("correct") is True
        if not submission_valid or not evaluated:
            task_outcome = "not_evaluated"
            score = None
            passed = None
        else:
            task_outcome = "correct" if correct else "incorrect"
            score = float(raw.get("score", 1.0 if correct else 0.0))
            passed = correct
        score_kind = str(raw.get("score_scope", "diagnostic"))
        if score_kind not in {"diagnostic", "subset"}:
            score_kind = "diagnostic"
        normalized = raw.get("normalized_answer", submission.answer)
        provenance = {
            "evaluator_id": (
                "offline_arena.adapter_private_gateway."
                + type(self._adapter).__name__
            ),
            "implementation_hash": _implementation_hash(self._adapter),
            "config_hash": hashlib.sha256(
                json.dumps(
                    {
                        "benchmark_id": task.benchmark_id,
                        "score_kind": score_kind,
                        "answer_format": task.answer_format,
                    },
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest(),
            "score_scope": score_kind,
            "official_evaluator_integrated": False,
            "reference_source_hash": None,
            "reference_access_stage": "post_submission_only",
        }
        self._evaluated = True
        return EvaluationResult(
            episode_id=episode_id,
            benchmark_id=task.benchmark_id,
            sample_id=task.sample_id,
            metric=str(raw.get("metric", "adapter_diagnostic")),
            score=score,
            passed=passed,
            prediction_normalized="" if normalized is None else str(normalized),
            task_outcome=task_outcome,
            evidence_refs=submission.evidence_refs,
            execution_success=True,
            contract_valid=True,
            submission_valid=submission_valid,
            score_kind=score_kind,
            official_evaluator_used=False,
            official_score=None,
            denominator_eligible=bool(raw.get("denominator_eligible", True)),
            error_type=(
                None
                if correct
                else "model_answer_error"
                if submission_valid and evaluated
                else "invalid_answer_format"
            ),
            notes="adapter private diagnostic; not an official benchmark score",
            evaluator_provenance=provenance,
        )

    def discard_reference(self) -> None:
        self._discarded = True


def _asset_source(adapter: BaseAdapter, asset_id: str) -> Path:
    resolver = getattr(adapter, "resolve_asset_path", None)
    if not callable(resolver):
        resolver = getattr(adapter, "resolve_asset", None)
    if not callable(resolver):
        raise TypeError("adapter does not expose a private asset resolver")
    path = Path(resolver(asset_id)).resolve()
    if not path.is_file():
        raise FileNotFoundError("adapter resolved an unreadable asset")
    return path


def materialize_typed_sample(
    adapter: BaseAdapter,
    sample_id: str | int,
    *,
    observation_policy: str,
) -> tuple[AdapterSample, PublicTask, tuple[AssetDescriptor, ...], dict[str, Path]]:
    """Materialize one row without allowing a path into model-visible contracts."""

    sample = adapter.load_sample(sample_id)
    materialized = adapter.materialize_sample(sample.source_sample_id)
    if not isinstance(materialized, Mapping):
        raise TypeError("adapter materialization must be an object")
    raw_task = materialized.get("public_task")
    raw_assets = materialized.get("assets")
    if not isinstance(raw_task, Mapping) or not isinstance(raw_assets, Sequence):
        raise TypeError("adapter materialization is missing typed public components")

    task_value = dict(raw_task)
    task_value["observation_policy"] = observation_policy
    task = PublicTask.from_dict(task_value)
    assets = tuple(AssetDescriptor.from_dict(item) for item in raw_assets)
    if set(task.asset_ids) != {item.asset_id for item in assets}:
        raise ValueError("adapter task and asset catalog disagree")
    sources = {
        descriptor.asset_id: _asset_source(adapter, descriptor.asset_id)
        for descriptor in assets
    }
    return sample, task, assets, sources


def _initial_plan(
    adapter: BaseAdapter,
    sample: AdapterSample,
    *,
    harness_profile: str,
    observation_policy: str,
) -> Mapping[str, Any] | None:
    if observation_policy == "active_selective_catalog_only":
        return None
    planner = getattr(adapter, "initial_observation_plan", None)
    if not callable(planner):
        return None
    plan_profile = "direct" if observation_policy == "full_context" else harness_profile
    try:
        plan = planner(sample.source_sample_id, profile=plan_profile)
    except Exception:
        raise TypeError("adapter initial observation plan failed") from None
    if not isinstance(plan, Mapping):
        raise TypeError("adapter initial observation plan must be an object")
    return plan


def _merge_plan_assets(
    adapter: BaseAdapter,
    task: PublicTask,
    assets: tuple[AssetDescriptor, ...],
    sources: dict[str, Path],
    plan: Mapping[str, Any] | None,
) -> tuple[PublicTask, tuple[AssetDescriptor, ...], dict[str, Path], tuple[str, ...]]:
    if plan is None:
        return task, assets, sources, ()
    raw_visible = plan.get("visible_assets")
    if not isinstance(raw_visible, Sequence) or isinstance(
        raw_visible, (str, bytes, bytearray)
    ):
        return task, assets, sources, ()
    merged = list(assets)
    known = {item.asset_id for item in merged}
    visible: list[str] = []
    for raw in raw_visible:
        if not isinstance(raw, Mapping):
            raise TypeError("adapter visible asset must be an object")
        descriptor = AssetDescriptor.from_dict(raw)
        visible.append(descriptor.asset_id)
        if descriptor.asset_id not in known:
            merged.append(descriptor)
            known.add(descriptor.asset_id)
            sources[descriptor.asset_id] = _asset_source(adapter, descriptor.asset_id)
    if not visible:
        return task, tuple(merged), sources, ()
    task = replace(task, asset_ids=tuple(item.asset_id for item in merged))
    return task, tuple(merged), sources, tuple(visible)


def materialize_environment_public_sample(
    adapter: BaseAdapter,
    sample_id: str | int,
    *,
    harness_profile: str,
    observation_policy: str,
) -> tuple[
    AdapterSample,
    PublicTask,
    tuple[AssetDescriptor, ...],
    dict[str, Path],
    tuple[str, ...],
]:
    """Return the exact public task/assets used by the sole environment builder."""

    sample, task, assets, sources = materialize_typed_sample(
        adapter,
        sample_id,
        observation_policy=observation_policy,
    )
    plan = _initial_plan(
        adapter,
        sample,
        harness_profile=harness_profile,
        observation_policy=observation_policy,
    )
    task, assets, sources, planned_visible = _merge_plan_assets(
        adapter,
        task,
        assets,
        sources,
        plan,
    )
    return sample, task, assets, sources, planned_visible


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _frame_requests(
    envelope: Any,
    *,
    maximum: int,
) -> tuple[dict[str, Any], ...]:
    arguments = dict(envelope.arguments)
    arguments.pop("asset_id", None)
    if envelope.action is ActionType.GET_FRAME:
        if "timestamp_ms" in arguments:
            timestamp_ms = _nonnegative_int(arguments["timestamp_ms"], "timestamp_ms")
            return ({"timestamp_seconds": timestamp_ms / 1000.0},)
        return ({"frame_index": _nonnegative_int(arguments["frame_index"], "frame_index")},)

    raw_indices = arguments.get("frame_indices")
    if raw_indices is not None:
        if set(arguments) - {"frame_indices"}:
            raise ValueError("frame_indices cannot be combined with window parameters")
        if isinstance(raw_indices, (str, bytes, bytearray)) or not isinstance(
            raw_indices, Sequence
        ):
            raise ValueError("frame_indices must be a sequence")
        indices = tuple(
            _nonnegative_int(value, "frame_index") for value in raw_indices
        )
        if not indices or indices != tuple(sorted(set(indices))):
            raise ValueError("frame_indices must be non-empty and strictly increasing")
    else:
        start = _nonnegative_int(arguments.get("start_frame", 0), "start_frame")
        step = _positive_int(arguments.get("step", 1), "step")
        end = arguments.get("end_frame")
        count = arguments.get("count")
        if end is not None and count is not None:
            raise ValueError("end_frame and count are mutually exclusive")
        if end is not None:
            stop = _nonnegative_int(end, "end_frame")
            if stop <= start:
                raise ValueError("end_frame must be greater than start_frame")
            indices = tuple(range(start, stop, step))
        else:
            requested = 3 if count is None else _positive_int(count, "count")
            indices = tuple(start + offset * step for offset in range(requested))
    if len(indices) > maximum:
        raise ValueError("frame window exceeds the configured limit")
    return tuple({"frame_index": index} for index in indices)


def _adapter_frame_record(
    transport: MediaTransport,
    adapter: BaseAdapter,
    descriptor: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    operation: str,
) -> MediaTransportRecord:
    typed = AssetDescriptor.from_dict(descriptor)
    source = _asset_source(adapter, typed.asset_id)
    opened = transport.open_asset(source, expected_sha256=typed.source_media_hash)
    index = typed.metadata.get("source_frame_index")
    if isinstance(index, bool) or not isinstance(index, int):
        index = typed.metadata.get("frozen_frame_ordinal")
    if isinstance(index, bool) or not isinstance(index, int):
        index = metadata.get("source_frame_index")
    if isinstance(index, bool) or not isinstance(index, int):
        index = metadata.get("frozen_frame_ordinal")
    if isinstance(index, bool) or not isinstance(index, int):
        index = None
    timestamp = metadata.get("timestamp_seconds")
    timestamp_ms = None
    if isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool):
        timestamp_ms = int(round(float(timestamp) * 1000.0))
    identity = {
        "operation": operation,
        "adapter_asset_id": typed.asset_id,
        "content_sha256": opened.content_sha256,
        "frame_index": index,
        "timestamp_ms": timestamp_ms,
    }
    derivation = hashlib.sha256(
        json.dumps(
            identity,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return replace(
        opened,
        operation=operation,
        asset_id=f"asset-{derivation[:32]}",
        derivation_sha256=derivation,
        parent_asset_ids=(typed.asset_id,),
        frame_id=f"frame-{derivation[:32]}",
        frame_index=index,
        timestamp_ms=timestamp_ms,
        frame_count=None,
        duration_ms=None,
        fps_numerator=None,
        fps_denominator=None,
    )


def _adapter_frame_resolver(
    adapter: BaseAdapter,
    sample: AdapterSample,
    transport: MediaTransport,
    known_asset_ids: frozenset[str],
):
    executor = getattr(adapter, "execute_action", None)
    if not callable(executor):
        return None

    def resolve(envelope: Any) -> MediaTransportRecord | Sequence[MediaTransportRecord]:
        anchor = envelope.arguments.get("asset_id")
        if anchor not in known_asset_ids:
            raise OfflineArenaContractError("frame action references an unknown asset")
        try:
            requests = _frame_requests(
                envelope,
                maximum=transport.max_frame_window,
            )
            records: list[MediaTransportRecord] = []
            for request in requests:
                raw = executor(sample.source_sample_id, "GET_FRAME", request)
                if not isinstance(raw, Mapping):
                    raise TypeError("adapter frame action returned a non-object")
                descriptor = raw.get("asset")
                metadata = raw.get("frame")
                if not isinstance(descriptor, Mapping) or not isinstance(
                    metadata, Mapping
                ):
                    raise TypeError("adapter frame action omitted typed media")
                records.append(
                    _adapter_frame_record(
                        transport,
                        adapter,
                        descriptor,
                        metadata,
                        operation=envelope.action.value,
                    )
                )
        except OfflineArenaContractError:
            raise
        except Exception:
            raise OfflineArenaContractError("adapter frame action failed") from None
        return records[0] if envelope.action is ActionType.GET_FRAME else records

    return resolve


def build_offline_environment(
    adapter: BaseAdapter,
    sample_id: str | int,
    *,
    harness_profile: str,
    observation_policy: str,
    media_transport: MediaTransport,
    checkpoint_path: str | Path | None = None,
    event_sink: Callable[[Mapping[str, Any]], None] | None = None,
    media_intervention: MediaInterventionPlan | None = None,
    modality_evidence_bundle: ModalityEvidenceBundle | None = None,
    evidence_sufficiency_requirement: (
        EvidenceSufficiencyRequirement | None
    ) = None,
    fixed_initial_media_records: Sequence[MediaTransportRecord] | None = None,
) -> OfflineEvidenceEnvironment:
    """Build the sole CPU environment for direct, full, or selective evidence."""

    if harness_profile not in {"direct", "w2_light"}:
        raise ValueError("harness_profile must be direct or w2_light")
    if observation_policy not in {
        "full_context",
        "selective_evidence",
        "active_selective_catalog_only",
    }:
        raise ValueError("unsupported observation_policy")
    if modality_evidence_bundle is not None and not isinstance(
        modality_evidence_bundle, ModalityEvidenceBundle
    ):
        raise TypeError(
            "modality_evidence_bundle must be ModalityEvidenceBundle"
        )
    if evidence_sufficiency_requirement is not None and not isinstance(
        evidence_sufficiency_requirement, EvidenceSufficiencyRequirement
    ):
        raise TypeError(
            "evidence_sufficiency_requirement must be "
            "EvidenceSufficiencyRequirement"
        )
    if modality_evidence_bundle is not None and harness_profile != "direct":
        raise ValueError("direct evidence bundle requires harness_profile=direct")
    if fixed_initial_media_records is not None and modality_evidence_bundle is None:
        raise ValueError(
            "fixed initial media records require a modality evidence bundle"
        )
    if evidence_sufficiency_requirement is not None and (
        harness_profile != "w2_light"
        or observation_policy != "active_selective_catalog_only"
    ):
        raise ValueError(
            "typed sufficiency requirement requires active catalog-only W2-light"
        )
    if media_intervention is not None and (
        modality_evidence_bundle is not None
        or evidence_sufficiency_requirement is not None
    ):
        raise ValueError(
            "visual sufficiency binding cannot be combined with media_intervention"
        )
    if media_intervention is not None:
        if not isinstance(media_intervention, MediaInterventionPlan):
            raise TypeError("media_intervention must be MediaInterventionPlan")
        media_intervention.validate_execution(
            harness_profile=harness_profile,
            observation_policy=observation_policy,
        )
    sample, task, assets, sources, planned_visible = (
        materialize_environment_public_sample(
            adapter,
            sample_id,
            harness_profile=harness_profile,
            observation_policy=observation_policy,
        )
    )
    resolved_intervention: ResolvedMediaIntervention | None = None
    if media_intervention is not None:
        if (
            media_intervention.target_benchmark_id != task.benchmark_id
            or media_intervention.target_sample_id != sample.source_sample_id
        ):
            raise MediaInterventionError(
                "media intervention does not match the materialized target"
            )
        if media_intervention.condition_id in {"M0", "M1", "M2"}:
            donor_sample: AdapterSample | None = None
            donor_task: PublicTask | None = None
            donor_assets: tuple[AssetDescriptor, ...] = ()
            donor_sources: dict[str, Path] | None = None
            donor_visible: tuple[str, ...] = ()
            if media_intervention.condition_id == "M2":
                donor_sample_id = media_intervention.donor_sample_id
                if donor_sample_id is None:  # validate() normally catches this.
                    raise MediaInterventionError("M2 donor sample is missing")
                donor_sample, donor_task, donor_assets, donor_sources = (
                    materialize_typed_sample(
                        adapter,
                        donor_sample_id,
                        observation_policy=observation_policy,
                    )
                )
                donor_plan = _initial_plan(
                    adapter,
                    donor_sample,
                    harness_profile=harness_profile,
                    observation_policy=observation_policy,
                )
                (
                    donor_task,
                    donor_assets,
                    donor_sources,
                    donor_planned_visible,
                ) = _merge_plan_assets(
                    adapter,
                    donor_task,
                    donor_assets,
                    donor_sources,
                    donor_plan,
                )
                donor_visible = donor_planned_visible or donor_task.asset_ids
            resolved_intervention = media_intervention.resolve(
                target_source_sample_id=sample.source_sample_id,
                target_task=task,
                target_assets=assets,
                target_sources=sources,
                target_visible_asset_ids=planned_visible or task.asset_ids,
                donor_source_sample_id=(
                    None
                    if donor_sample is None
                    else donor_sample.source_sample_id
                ),
                donor_task=donor_task,
                donor_assets=donor_assets,
                donor_sources=donor_sources,
                donor_visible_asset_ids=donor_visible,
            )
            sources.update(resolved_intervention.source_bindings)
    if harness_profile == "direct":
        actions = (ActionType.SUBMIT,)
        initial_visible = (
            modality_evidence_bundle.source_asset_ids
            if modality_evidence_bundle is not None
            else resolved_intervention.initial_visible_asset_ids
            if resolved_intervention is not None
            else planned_visible or task.asset_ids
        )
        max_actions = 1
        media_ceiling = 0
    else:
        if evidence_sufficiency_requirement is not None:
            actions = _sufficiency_actions(evidence_sufficiency_requirement)
        else:
            action_map = (
                _ACTIVE_SELECTIVE_ACTIONS_BY_BENCHMARK
                if observation_policy == "active_selective_catalog_only"
                else _ACTIONS_BY_BENCHMARK
            )
            try:
                actions = action_map[task.benchmark_id]
            except KeyError:
                raise ValueError(
                    "adapter benchmark is not in the six-source registry"
                ) from None
        if observation_policy == "active_selective_catalog_only":
            initial_visible = ()
        else:
            initial_visible = planned_visible or (
                task.asset_ids
                if observation_policy == "full_context"
                else task.asset_ids[:1]
            )
        if media_intervention is None:
            max_actions = 8
            media_ceiling = 6
        else:
            max_actions = 12
            media_ceiling = 8
    policy_kwargs: dict[str, Any] = {
        "allowed_actions": actions,
        "max_actions": max_actions,
        "media_operation_ceiling": media_ceiling,
        "current_profile": harness_profile,
    }
    if "submit_precondition" in getattr(ActionPolicy, "__dataclass_fields__", {}):
        policy_kwargs["submit_precondition"] = (
            MODALITY_SUFFICIENT_EVIDENCE
            if evidence_sufficiency_requirement is not None
            else None
            if media_intervention is None
            else media_intervention.definition.submit_precondition
        )
    policy = ActionPolicy(**policy_kwargs)
    evaluator = AdapterPrivateEvaluatorGateway(adapter, sample)
    environment_kwargs: dict[str, Any] = {
        "evaluator": evaluator,
        "action_policy": policy,
        "media_transport": media_transport,
        "asset_sources": sources,
        "checkpoint_path": checkpoint_path,
        "profile": harness_profile,
        "initial_visible_asset_ids": initial_visible,
        "event_sink": event_sink,
        "media_action_resolver": _adapter_frame_resolver(
            adapter,
            sample,
            media_transport,
            frozenset(task.asset_ids),
        ),
        "modality_evidence_bundle": modality_evidence_bundle,
        "evidence_sufficiency_requirement": evidence_sufficiency_requirement,
        "fixed_initial_media_records": fixed_initial_media_records,
    }
    if (
        resolved_intervention is not None
        and resolved_intervention.source_content_sha256_overrides
    ):
        # Private integrity override required by M2 while the public descriptor
        # remains the target catalog.  OfflineEvidenceEnvironment owns this
        # private argument and must never project it into snapshots/observations.
        environment_kwargs["source_content_sha256_overrides"] = (
            resolved_intervention.source_content_sha256_overrides
        )
    return OfflineEvidenceEnvironment(task, assets, **environment_kwargs)


__all__ = [
    "AdapterPrivateEvaluatorGateway",
    "build_offline_environment",
    "materialize_environment_public_sample",
    "materialize_typed_sample",
]
