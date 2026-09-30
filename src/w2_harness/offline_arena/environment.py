"""CPU-only evidence environment with a strict public/private boundary."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from .contracts import (
    ActionEnvelope,
    ActionPolicy,
    ActionType,
    AssetDescriptor,
    EvaluationResult,
    EvidenceRecord,
    MEDIA_OBSERVATION_REQUIRED_BEFORE_SUBMIT,
    MODALITY_SUFFICIENT_EVIDENCE,
    Observation,
    OfflineArenaContractError,
    ONE_SUCCESSFUL_ANSWER_BEARING_MEDIA_OBSERVATION,
    ParsedSubmission,
    PublicTask,
    STATE_SCHEMA_VERSION,
    json_copy,
)
from .evaluator import PrivateEvaluationError, PrivateEvaluator
from .media_transport import MediaTransport, MediaTransportError, MediaTransportRecord
from .state import (
    AtomicCheckpointError,
    AtomicCheckpointStore,
    EpisodeLifecycle,
    EpisodeLifecycleState,
    EpisodeState,
    StateTransitionError,
    canonical_sha256,
)
from .visible_evidence import (
    EVIDENCE_BINDING_CONTRACT_HASH,
    VisiblePublicEvidenceBinding,
    assert_collision_safe,
)
from .visual_sufficiency import (
    EvidenceSufficiencyRequirement,
    ModalityEvidenceBundle,
    VisualModality,
    evaluate_evidence_sufficiency,
    summarize_public_asset_descriptors,
)


_MISSING = object()
_MEDIA_ACTIONS = frozenset(
    {
        ActionType.OPEN_ASSET,
        ActionType.GET_VIEW,
        ActionType.GET_FRAME,
        ActionType.GET_FRAME_WINDOW,
        ActionType.CROP_REGION,
        ActionType.ZOOM_REGION,
        ActionType.COMPOSE_ASSETS,
    }
)
_TEMPORAL_MODEL_MEDIA_FIELDS = frozenset(
    {
        "operation",
        "asset_id",
        "media_kind",
        "mime_type",
        "width",
        "height",
        "byte_length",
        "payload_bytes",
        "source_sha256",
        "content_sha256",
        "derivation_sha256",
        "parent_asset_ids",
        "data_url",
        "view_id",
        "frame_id",
    }
)
_PRIVATE_TEMPORAL_METADATA_PARTS = (
    "arm",
    "condition",
    "frame_index",
    "ordinal",
    "policy",
    "reference",
    "selection",
    "source_order",
    "timestamp",
    "true_order",
)


class OfflineArenaError(RuntimeError):
    """Base error for deterministic offline environment operations."""


class OfflineArenaLifecycleError(OfflineArenaError):
    """An operation was requested in the wrong lifecycle state."""


class ActionRejectedError(OfflineArenaError, ValueError):
    """A strict action was rejected and represented as typed feedback."""


class SubmissionRejectedError(ActionRejectedError):
    """A submission failed its public contract."""


class EvaluationFailedError(OfflineArenaError):
    """Private post-submission evaluation failed."""


class OfflineCheckpointError(OfflineArenaError):
    """The environment could not durably commit public episode state."""


def _episode_id(task: PublicTask) -> str:
    raw = f"offline:{task.benchmark_id}:{task.sample_id}"
    if len(raw) <= 192:
        return raw
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"offline:{task.sample_id[:150]}:{digest}"[:192]


def _checkpoint_target(path: str | Path, episode_id: str) -> Path:
    candidate = Path(path)
    if candidate.exists() and candidate.is_dir():
        return candidate / f"{episode_id}.checkpoint.json"
    if not candidate.suffix:
        return candidate / f"{episode_id}.checkpoint.json"
    return candidate


class OfflineEvidenceEnvironment:
    """One immutable sample plus mutable, content-addressed visible evidence."""

    profile_id = "offline_evidence"
    environment_profile = "offline_evidence"

    def __init__(
        self,
        task: PublicTask,
        asset_catalog: Sequence[AssetDescriptor | Mapping[str, Any]],
        private_truth: Any = _MISSING,
        *,
        evaluator: PrivateEvaluator | None = None,
        action_policy: ActionPolicy | None = None,
        media_transport: MediaTransport | None = None,
        asset_sources: Mapping[str, Any] | None = None,
        source_content_sha256_overrides: Mapping[str, str] | None = None,
        checkpoint_path: str | Path | None = None,
        episode_id: str | None = None,
        profile: str = "w2_light",
        initial_visible_asset_ids: Sequence[str] | None = None,
        event_sink: Callable[[Mapping[str, Any]], None] | None = None,
        media_action_resolver: (
            Callable[
                [ActionEnvelope],
                MediaTransportRecord | Sequence[MediaTransportRecord],
            ]
            | None
        ) = None,
        modality_evidence_bundle: ModalityEvidenceBundle | None = None,
        evidence_sufficiency_requirement: (
            EvidenceSufficiencyRequirement | None
        ) = None,
        fixed_initial_media_records: (
            Sequence[MediaTransportRecord] | None
        ) = None,
    ) -> None:
        if not isinstance(task, PublicTask):
            raise TypeError("task must be PublicTask")
        assets = tuple(
            item if isinstance(item, AssetDescriptor) else AssetDescriptor.from_dict(item)
            for item in asset_catalog
        )
        if len({item.asset_id for item in assets}) != len(assets):
            raise ValueError("asset catalog IDs must be unique")
        if set(task.asset_ids) != {item.asset_id for item in assets}:
            raise ValueError("task asset_ids and asset catalog disagree")
        if evaluator is not None and private_truth is not _MISSING:
            raise ValueError("provide either private_truth or evaluator, not both")
        if evaluator is not None and (
            not callable(getattr(evaluator, "evaluate", None))
            or not callable(getattr(evaluator, "discard_reference", None))
        ):
            raise TypeError("evaluator must implement the private evaluator gateway")
        if profile not in {"direct", "w2_light"}:
            raise ValueError("profile must be direct or w2_light")
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
        if modality_evidence_bundle is not None and profile != "direct":
            raise ValueError(
                "modality_evidence_bundle is only valid for direct initial media"
            )
        fixed_records: tuple[MediaTransportRecord, ...] | None = None
        if fixed_initial_media_records is not None:
            if modality_evidence_bundle is None:
                raise ValueError(
                    "fixed initial media requires a modality evidence bundle"
                )
            if isinstance(fixed_initial_media_records, (str, bytes, bytearray)):
                raise TypeError("fixed initial media records must be a sequence")
            fixed_records = tuple(fixed_initial_media_records)
            if not all(isinstance(item, MediaTransportRecord) for item in fixed_records):
                raise TypeError(
                    "fixed initial media records must be MediaTransportRecord values"
                )
            if len(fixed_records) != modality_evidence_bundle.presented_count:
                raise ValueError(
                    "fixed initial media count disagrees with the evidence bundle"
                )
        if evidence_sufficiency_requirement is not None and (
            profile != "w2_light"
            or task.observation_policy != "active_selective_catalog_only"
        ):
            raise ValueError(
                "evidence_sufficiency_requirement requires the active catalog-only profile"
            )
        if evidence_sufficiency_requirement is not None:
            summary = summarize_public_asset_descriptors(assets)
            requirement_public_facts = (
                evidence_sufficiency_requirement.modality,
                evidence_sufficiency_requirement.asset_count,
                evidence_sufficiency_requirement.frame_count,
                evidence_sufficiency_requirement.duration,
            )
            if requirement_public_facts != (
                summary.modality,
                summary.asset_count,
                summary.frame_count,
                summary.duration,
            ):
                raise ValueError(
                    "typed sufficiency requirement disagrees with public assets"
                )
        policy = action_policy or ActionPolicy(current_profile=profile)
        if policy.current_profile != profile:
            policy = replace(policy, current_profile=profile)
        if evidence_sufficiency_requirement is not None:
            if policy.submit_precondition is None:
                policy = replace(
                    policy,
                    submit_precondition=MODALITY_SUFFICIENT_EVIDENCE,
                )
            elif policy.submit_precondition != MODALITY_SUFFICIENT_EVIDENCE:
                raise ValueError(
                    "typed visual sufficiency requires its submit precondition"
                )
        elif policy.submit_precondition == MODALITY_SUFFICIENT_EVIDENCE:
            raise ValueError(
                "modality-sufficient submit precondition requires a typed requirement"
            )
        sources = dict(asset_sources or {})
        if set(sources) - set(task.asset_ids):
            raise ValueError("asset_sources contains an unknown asset ID")
        source_hash_overrides = dict(source_content_sha256_overrides or {})
        if set(source_hash_overrides) - set(task.asset_ids):
            raise ValueError(
                "source_content_sha256_overrides contains an unknown asset ID"
            )
        if any(
            not isinstance(digest, str)
            or len(digest) != 64
            or digest != digest.casefold()
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in source_hash_overrides.values()
        ):
            raise ValueError(
                "source_content_sha256_overrides values must be SHA-256 digests"
            )
        visible = tuple(initial_visible_asset_ids or ())
        direct_presentation_asset_ids: tuple[str, ...] | None = None
        if modality_evidence_bundle is not None:
            if initial_visible_asset_ids is not None and tuple(
                initial_visible_asset_ids
            ) != modality_evidence_bundle.source_asset_ids:
                raise ValueError(
                    "initial_visible_asset_ids disagree with the direct evidence bundle"
                )
            direct_presentation_asset_ids = (
                modality_evidence_bundle.source_asset_ids
            )
            # EpisodeState models which public assets have been revealed, so it
            # remains a set-like sequence.  The evidence bundle separately owns
            # presentation multiplicity and order (including the repeated-frame
            # control).
            visible = tuple(dict.fromkeys(direct_presentation_asset_ids))
            by_id = {item.asset_id: item for item in assets}
            for presentation_index, (asset_id, media_hash) in enumerate(zip(
                modality_evidence_bundle.source_asset_ids,
                modality_evidence_bundle.source_media_hashes,
                strict=True,
            )):
                descriptor = by_id.get(asset_id)
                if descriptor is None:
                    raise ValueError(
                        "direct evidence bundle references an unknown asset"
                    )
                if fixed_records is None:
                    expected_media_hash = descriptor.source_media_hash
                else:
                    record = fixed_records[presentation_index]
                    expected_media_hash = (
                        record.content_sha256
                        if modality_evidence_bundle.modality is VisualModality.VIDEO
                        else record.source_sha256
                    )
                if expected_media_hash != media_hash:
                    raise ValueError(
                        "direct evidence bundle media hash disagrees with its exact input"
                    )
        elif initial_visible_asset_ids is None and (
            profile == "direct" or "full_context" in task.observation_policy
        ):
            visible = task.asset_ids
        if not set(visible) <= set(task.asset_ids):
            raise ValueError("initial visible assets are not in the public catalog")

        self._task = task
        self._assets = {item.asset_id: item for item in assets}
        self._asset_sources = sources
        self._source_content_sha256_overrides = source_hash_overrides
        self._policy = policy
        self._profile = profile
        self._transport = media_transport
        self._episode_id = episode_id or _episode_id(task)
        self._initial_visible = visible
        self._direct_presentation_asset_ids = direct_presentation_asset_ids
        self._event_sink = event_sink
        self._media_action_resolver = media_action_resolver
        self._modality_evidence_bundle = modality_evidence_bundle
        self._fixed_initial_media_records = fixed_records
        self._evidence_sufficiency_requirement = evidence_sufficiency_requirement
        self._visual_sufficiency_binding = (
            modality_evidence_bundle is not None
            or evidence_sufficiency_requirement is not None
        )
        self._events: list[dict[str, Any]] = []
        self.__evaluator = evaluator or PrivateEvaluator(
            None if private_truth is _MISSING else private_truth
        )
        self._lifecycle = EpisodeLifecycle()
        self._revealed_assets: list[str] = []
        self._derived_assets: dict[str, AssetDescriptor] = {}
        self._media_records: dict[str, MediaTransportRecord] = {}
        self._media_root_bindings: dict[str, tuple[dict[str, Any], ...]] = {}
        self._media_root_operations: dict[str, str] = {}
        self._pending_root_presentations: tuple[dict[str, Any], ...] = ()
        self._observed_root_keys: set[tuple[str, str, int | None]] = set()
        self._observed_evidence_bundles: list[ModalityEvidenceBundle] = []
        self._evidence: dict[str, EvidenceRecord] = {}
        self._provisional_bindings: dict[
            str, VisiblePublicEvidenceBinding
        ] = {}
        self._pending_visibility_refs: tuple[str, ...] = ()
        self._provisional_model_visible_media: tuple[dict[str, Any], ...] = ()
        self._model_visible_media: tuple[dict[str, Any], ...] = ()
        self._last_visibility_disposition: dict[str, Any] | None = None
        self._action_history: list[dict[str, Any]] = []
        self._semantic_cache: dict[str, dict[str, Any]] = {}
        self._environment_action_count = 0
        self._media_action_count = 0
        self._model_call_count = 0
        self._submission: ParsedSubmission | None = None
        self._evaluation: EvaluationResult | None = None
        self._last_new_evidence: tuple[EvidenceRecord, ...] = ()
        self._last_action_result: dict[str, Any] = {"status": "created"}
        self._last_safe_error: dict[str, Any] | None = None
        self._restored_from_checkpoint = False
        self._checkpoint_store = (
            None
            if checkpoint_path is None
            else AtomicCheckpointStore(
                _checkpoint_target(checkpoint_path, self._episode_id)
            )
        )
        if (
            fixed_records is not None
            and self._checkpoint_store is not None
            and self._checkpoint_store.path.exists()
        ):
            self._restore_active_checkpoint()

    @property
    def episode_id(self) -> str:
        return self._episode_id

    @property
    def state(self) -> EpisodeLifecycleState:
        return self._lifecycle.state

    @property
    def checkpoint_path(self) -> Path | None:
        return None if self._checkpoint_store is None else self._checkpoint_store.path

    @property
    def resumed_from_checkpoint(self) -> bool:
        return self._restored_from_checkpoint

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        return tuple(json_copy(self._events, "events"))

    def private_identity_commitments(self) -> dict[str, str]:
        """Return execution-only content commitments, never a model projection."""

        commitments: dict[str, str] = {}
        if self._modality_evidence_bundle is not None:
            commitments["modality_evidence_bundle_sha256"] = (
                self._modality_evidence_bundle.canonical_hash
            )
        if self._evidence_sufficiency_requirement is not None:
            commitments["evidence_sufficiency_requirement_sha256"] = (
                self._evidence_sufficiency_requirement.canonical_hash
            )
        return commitments

    @property
    def episode_state(self) -> EpisodeState:
        core = {
            "episode_id": self._episode_id,
            "benchmark_id": self._task.benchmark_id,
            "sample_id": self._task.sample_id,
            "profile": self._profile,
            "revision": self._lifecycle.revision,
            "revealed_assets": list(self._revealed_assets),
            "derived_assets": list(self._derived_assets),
            "evidence_refs": list(self._evidence),
            "action_history": list(self._action_history),
            "model_call_count": self._model_call_count,
            "environment_action_count": self._environment_action_count,
            "submission_status": self._submission_status(),
            "remaining_budget": self._remaining_budget(),
        }
        return EpisodeState(**core, checkpoint_hash=canonical_sha256(core))

    def reset(
        self,
        sample_id: str | None = None,
        profile: str | None = None,
    ) -> Observation:
        if sample_id is not None and sample_id != self._task.sample_id:
            raise OfflineArenaLifecycleError("reset sample_id does not match bound sample")
        if profile is not None and profile != self._profile:
            raise OfflineArenaLifecycleError("reset profile does not match bound profile")
        if self._restored_from_checkpoint:
            return self._initial_observation()
        self._require_state(EpisodeLifecycleState.CREATED)
        self._revealed_assets = list(self._initial_visible)
        self._lifecycle.transition(EpisodeLifecycleState.ACTIVE)
        self._last_action_result = {
            "status": "reset",
            "revealed": list(self._revealed_assets),
        }
        self._emit("EpisodeStarted", {"profile": self._profile})
        self._emit("StateUpdated", {"state_revision": self._lifecycle.revision})
        self._commit()
        return self._observation()

    def observe(self) -> Observation:
        if self.state in {EpisodeLifecycleState.CREATED, EpisodeLifecycleState.CLOSED}:
            raise OfflineArenaLifecycleError(
                f"observe is unavailable in state {self.state.value}"
            )
        return self._observation(new_evidence=())

    def record_model_call(self) -> None:
        self._require_state(EpisodeLifecycleState.ACTIVE)
        self._model_call_count += 1
        self._commit()

    def reject_action(
        self,
        error_type: str,
        *,
        message: str = "action rejected",
    ) -> Observation:
        """Persist parser feedback without treating it as an environment action."""

        self._require_state(EpisodeLifecycleState.ACTIVE)
        self._lifecycle.touch()
        self._last_new_evidence = ()
        self._last_safe_error = {
            "error_type": str(error_type)[:128],
            "message": str(message)[:256],
            "retryable_with_new_model_turn": True,
        }
        self._last_action_result = {"status": "rejected", "executed": False}
        self._emit("ActionRejected", self._last_safe_error)
        self._commit()
        return self._observation(new_evidence=())

    def step(self, action: ActionEnvelope | Mapping[str, Any] | str) -> Observation:
        self._require_state(EpisodeLifecycleState.ACTIVE)
        try:
            envelope = self._coerce_action(action)
            self._policy.validate(envelope)
            if self._environment_action_count >= self._policy.max_actions:
                return self._block("environment_action_budget_exhausted")
            if envelope.action in _MEDIA_ACTIONS and (
                self._media_action_count >= self._policy.media_operation_ceiling
            ):
                return self._reject_envelope(
                    envelope, "media_operation_ceiling_exhausted"
                )
        except OfflineArenaContractError as exc:
            return self._reject_unparsed(type(exc).__name__)

        semantic = canonical_sha256(envelope.to_dict())
        cached = self._semantic_cache.get(semantic)
        if cached is not None:
            return self._record_cached(envelope, semantic, cached)
        if envelope.action is ActionType.SUBMIT:
            return self._submit_envelope(envelope, semantic)
        try:
            if envelope.action in _MEDIA_ACTIONS:
                return self._media_step(envelope, semantic)
            if envelope.action is ActionType.RECORD_EVIDENCE:
                return self._record_evidence(envelope, semantic)
            return self._query_step(envelope, semantic)
        except (MediaTransportError, OfflineArenaContractError) as exc:
            return self._reject_envelope(envelope, type(exc).__name__)

    def submit(self, payload: Mapping[str, Any]) -> Observation:
        if isinstance(payload, Mapping) and set(payload) == {
            "answer",
            "evidence_refs",
            "confidence",
        }:
            payload = {"action": "SUBMIT", "arguments": dict(payload)}
        return self.step(payload)

    def evaluate(self) -> EvaluationResult:
        if self.state is EpisodeLifecycleState.EVALUATED and self._evaluation is not None:
            return self._evaluation
        self._require_state(EpisodeLifecycleState.SUBMITTED)
        if self._submission is None:
            raise EvaluationFailedError("submitted state has no submission")
        try:
            result = self.__evaluator.evaluate(
                self._submission,
                state=self.state,
                task=self._task,
                episode_id=self._episode_id,
            )
        except PrivateEvaluationError:
            self.__evaluator.discard_reference()
            self._lifecycle.transition(
                EpisodeLifecycleState.FAILED,
                failure_code="evaluator_error",
            )
            self._commit()
            raise EvaluationFailedError("private evaluator failed") from None
        self._evaluation = result
        self._lifecycle.transition(EpisodeLifecycleState.EVALUATED)
        self.__evaluator.discard_reference()
        self._last_action_result = {
            "status": "evaluated",
            "task_outcome": result.task_outcome,
        }
        self._emit("EvaluationCompleted", {"task_outcome": result.task_outcome})
        self._emit("StateUpdated", {"state_revision": self._lifecycle.revision})
        self._commit()
        return result

    def close(self) -> Observation:
        if self.state is EpisodeLifecycleState.CLOSED:
            return self._observation(new_evidence=())
        self.__evaluator.discard_reference()
        try:
            self._lifecycle.transition(EpisodeLifecycleState.CLOSED)
        except StateTransitionError as exc:
            raise OfflineArenaLifecycleError(str(exc)) from None
        self._last_action_result = {"status": "closed"}
        self._emit("EpisodeTerminated", {"status": "CLOSED"})
        self._emit("StateUpdated", {"state_revision": self._lifecycle.revision})
        self._commit()
        return self._observation(new_evidence=())

    def model_media(
        self,
        _observation: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Prepare public payloads and stable, not-yet-committed evidence refs."""

        if self._transport is None:
            self.discard_provisional_model_visible_media("transport_unavailable")
            return []
        prepared: list[tuple[str, AssetDescriptor, MediaTransportRecord]] = []
        assets = self._all_assets()
        presentation_asset_ids = (
            self._direct_presentation_asset_ids
            if self._direct_presentation_asset_ids is not None
            else tuple(self._revealed_assets)
        )
        for presentation_index, public_asset_id in enumerate(
            presentation_asset_ids
        ):
            descriptor = assets.get(public_asset_id)
            if descriptor is None:
                continue
            if self._fixed_initial_media_records is not None:
                prepared.append(
                    (
                        public_asset_id,
                        descriptor,
                        self._fixed_initial_media_records[presentation_index],
                    )
                )
                continue
            if public_asset_id in self._media_records:
                prepared.append(
                    (public_asset_id, descriptor, self._media_records[public_asset_id])
                )
                continue
            source = self._asset_sources.get(public_asset_id)
            if source is None:
                continue
            if descriptor.modality == "video":
                video_source: Any = source
                source_hash_override = self._source_content_sha256_overrides.get(
                    public_asset_id
                )
                if source_hash_override is not None:
                    video_source = {
                        "source": source,
                        "expected_sha256": source_hash_override,
                    }
                prepared.extend(
                    (public_asset_id, descriptor, record)
                    for record in self._transport.full_context_records([video_source])
                )
            elif descriptor.source_kind == "view":
                prepared.append(
                    (
                        public_asset_id,
                        descriptor,
                        self._transport.get_view(
                            public_asset_id,
                            source=source,
                            expected_sha256=self._expected_source_hash(
                                public_asset_id
                            ),
                        ),
                    )
                )
            else:
                prepared.append(
                    (
                        public_asset_id,
                        descriptor,
                        self._transport.open_asset(
                            source,
                            expected_sha256=self._expected_source_hash(
                                public_asset_id
                            ),
                        ),
                    )
                )

        provisional: dict[str, VisiblePublicEvidenceBinding] = {}
        media: list[dict[str, Any]] = []
        projections: list[dict[str, Any]] = []
        root_presentations: list[dict[str, Any]] = []
        for public_asset_id, descriptor, record in prepared:
            binding = VisiblePublicEvidenceBinding.from_media(
                benchmark_id=self._task.benchmark_id,
                sample_id=self._task.sample_id,
                public_asset_id=public_asset_id,
                descriptor=descriptor,
                record=record,
            )
            evidence_id = binding.evidence.evidence_id
            assert_collision_safe(self._evidence.get(evidence_id), binding.evidence)
            prior = provisional.get(evidence_id)
            if prior is not None and prior.evidence != binding.evidence:
                raise OfflineArenaContractError(
                    "one evidence identity maps to divergent media bindings"
                )
            provisional.setdefault(evidence_id, binding)
            item = record.to_model_visible_dict()
            if self._visual_video_binding():
                item = {
                    key: value
                    for key, value in item.items()
                    if key in _TEMPORAL_MODEL_MEDIA_FIELDS
                }
            item.update(binding.media_fields())
            if self._visual_sufficiency_binding:
                item["sequence_position"] = len(media)
            media.append(item)
            projections.append(binding.observation_projection())
            roots = self._media_roots_for_visibility(
                public_asset_id,
                descriptor,
                record,
            )
            if roots:
                root_presentations.append(
                    {
                        "operation": self._media_root_operations.get(
                            record.asset_id,
                            record.operation,
                        ),
                        "roots": [dict(root) for root in roots],
                        "evidence_ref": evidence_id,
                        "payload_bytes": record.payload_bytes,
                    }
                )

        self._provisional_bindings = provisional
        self._pending_visibility_refs = tuple(provisional)
        self._provisional_model_visible_media = tuple(projections)
        self._pending_root_presentations = tuple(root_presentations)
        return media

    def project_model_visible_media(
        self,
        observation: Mapping[str, Any],
        media: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Project exact provisional handles into the request observation."""

        self._require_state(EpisodeLifecycleState.ACTIVE)
        refs: list[str] = []
        for item in media:
            ref = item.get("evidence_ref")
            if not isinstance(ref, str) or ref not in self._pending_visibility_refs:
                raise OfflineArenaContractError(
                    "model media lacks its exact provisional evidence ref"
                )
            refs.append(ref)
        if tuple(dict.fromkeys(refs)) != self._pending_visibility_refs:
            raise OfflineArenaContractError(
                "model media and provisional evidence bindings disagree"
            )
        projected = json_copy(
            observation,
            "projected observation",
            public_task=True,
        )
        projected["valid_submission_evidence_refs"] = list(
            dict.fromkeys((*self._evidence, *self._pending_visibility_refs))
        )
        projected["model_visible_media"] = list(
            self._provisional_model_visible_media
        )
        projected["answer_bearing_media_visible"] = bool(media)
        projected[
            "evidence_binding_contract_hash"
        ] = EVIDENCE_BINDING_CONTRACT_HASH
        return projected

    def commit_model_visible_media(
        self,
        *,
        request_id: str,
        cache_hit: bool,
    ) -> Observation:
        """Atomically promote the current request's handles after a response."""

        self._require_state(EpisodeLifecycleState.ACTIVE)
        pending_refs = self._pending_visibility_refs
        existing_refs = tuple(
            evidence_id
            for evidence_id in pending_refs
            if evidence_id in self._evidence
        )
        if pending_refs and self._visual_sufficiency_binding:
            self._emit_evidence_lifecycle_stage(
                "delivered",
                pending_refs,
                request_id=request_id,
            )
            self._emit_evidence_lifecycle_stage(
                "observed",
                pending_refs,
                request_id=request_id,
            )
        created: list[EvidenceRecord] = []
        for evidence_id in self._pending_visibility_refs:
            binding = self._provisional_bindings[evidence_id]
            existing = self._evidence.get(evidence_id)
            assert_collision_safe(existing, binding.evidence)
            if existing is None:
                self._evidence[evidence_id] = binding.evidence
                created.append(binding.evidence)
                created_payload: dict[str, Any] = {"evidence_id": evidence_id}
                if self._visual_sufficiency_binding:
                    created_payload["evidence_lifecycle_stage"] = "created"
                self._emit("EvidenceCreated", created_payload)
        self._record_observed_root_bundles()
        if existing_refs and self._visual_sufficiency_binding:
            self._emit_evidence_lifecycle_stage(
                "reentered",
                existing_refs,
                request_id=request_id,
            )
        if created:
            self._lifecycle.touch()
        self._last_new_evidence = tuple(created)
        self._model_visible_media = self._provisional_model_visible_media
        self._emit(
            "EvidenceVisibilityCommitted",
            {
                "request_id": request_id,
                "evidence_refs": list(self._pending_visibility_refs),
                "media_count": len(self._provisional_model_visible_media),
                "cache_hit": bool(cache_hit),
                "binding_contract_hash": EVIDENCE_BINDING_CONTRACT_HASH,
            },
        )
        if created:
            self._emit("StateUpdated", {"state_revision": self._lifecycle.revision})
        self._provisional_bindings = {}
        self._pending_visibility_refs = ()
        self._provisional_model_visible_media = ()
        self._pending_root_presentations = ()
        self._commit()
        return self._observation(new_evidence=tuple(created))

    def discard_provisional_model_visible_media(self, disposition: str) -> None:
        """Discard uncommitted refs after a request that produced no response."""

        if not isinstance(disposition, str) or not disposition:
            raise ValueError("visibility disposition must be non-empty")
        pending_count = len(self._pending_visibility_refs)
        if pending_count:
            status = (
                "VISIBILITY_UNKNOWN_NO_ACTION"
                if disposition == "no_terminal_response"
                else "NOT_MODEL_VISIBLE"
            )
            self._last_visibility_disposition = {
                "status": status,
                "reason": disposition[:128],
                "provisional_ref_count": pending_count,
                "committed_ref_count": 0,
            }
            self._lifecycle.touch()
            self._emit(
                "StateUpdated",
                {
                    "state_revision": self._lifecycle.revision,
                    "evidence_visibility_disposition": status,
                    "provisional_ref_count": pending_count,
                },
            )
        self._provisional_bindings = {}
        self._pending_visibility_refs = ()
        self._provisional_model_visible_media = ()
        self._pending_root_presentations = ()
        if pending_count:
            self._commit()

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "private_reference_access_policy": "post_submission_only",
            "lifecycle": self._lifecycle.to_dict(),
            "episode_state": self.episode_state.to_dict(),
            "task": self._task.to_dict(),
            "asset_catalog": [item.to_dict() for item in self._assets.values()],
            "action_policy": replace(
                self._policy,
                state_revision=self._lifecycle.revision,
            ).to_dict(),
            "evidence": [item.to_dict() for item in self._evidence.values()],
            "model_visible_media": list(self._model_visible_media),
            "evidence_binding_contract_hash": EVIDENCE_BINDING_CONTRACT_HASH,
            "last_visibility_disposition": (
                None
                if self._last_visibility_disposition is None
                else json_copy(
                    self._last_visibility_disposition,
                    "last_visibility_disposition",
                )
            ),
            "submission": (
                None if self._submission is None else self._submission.to_dict()
            ),
            "evaluation": (
                None if self._evaluation is None else self._evaluation.to_dict()
            ),
            "last_action_result": json_copy(
                self._last_action_result, "action_result"
            ),
            "last_safe_error": (
                None
                if self._last_safe_error is None
                else json_copy(self._last_safe_error, "safe_error")
            ),
        }

    def _restore_active_checkpoint(self) -> None:
        assert self._checkpoint_store is not None
        try:
            payload = self._checkpoint_store.load()
            lifecycle = EpisodeLifecycle.from_dict(payload["lifecycle"])
            expected = {
                "schema_version": STATE_SCHEMA_VERSION,
                "task": self._task.to_dict(),
                "asset_catalog": [item.to_dict() for item in self._assets.values()],
                "action_policy": replace(
                    self._policy, state_revision=lifecycle.revision
                ).to_dict(),
                "private_reference_access_policy": "post_submission_only",
                "evidence_binding_contract_hash": EVIDENCE_BINDING_CONTRACT_HASH,
                "submission": None,
                "evaluation": None,
            }
            if lifecycle.state is not EpisodeLifecycleState.ACTIVE or any(
                payload.get(key) != value for key, value in expected.items()
            ):
                raise ValueError("environment checkpoint binding mismatch")

            raw_episode = payload["episode_state"]
            episode = EpisodeState(**dict(raw_episode))
            episode_body = episode.to_dict()
            recorded_hash = episode_body.pop("checkpoint_hash")
            if (
                episode.to_dict() != raw_episode
                or recorded_hash != canonical_sha256(episode_body)
                or episode.episode_id != self._episode_id
                or episode.benchmark_id != self._task.benchmark_id
                or episode.sample_id != self._task.sample_id
                or episode.profile != self._profile
                or episode.revision != lifecycle.revision
                or episode.submission_status != "not_submitted"
                or episode.derived_assets
            ):
                raise ValueError("environment checkpoint identity mismatch")

            evidence_records = tuple(
                EvidenceRecord.from_dict(item) for item in payload["evidence"]
            )
            if (
                tuple(item.evidence_id for item in evidence_records)
                != episode.evidence_refs
                or any(item.source_kind == "private_truth" for item in evidence_records)
                or any(
                    item.asset_id is not None and item.asset_id not in self._assets
                    for item in evidence_records
                )
            ):
                raise ValueError("environment checkpoint evidence mismatch")
            evidence = {item.evidence_id: item for item in evidence_records}
            if len(evidence) != len(evidence_records):
                raise ValueError("environment checkpoint evidence collision")
            expected_visible = self._validate_fixed_evidence(evidence_records)

            model_visible = json_copy(
                payload["model_visible_media"],
                "checkpoint model-visible media",
                public_task=True,
            )
            if not isinstance(model_visible, list) or any(
                not isinstance(item, Mapping)
                or item.get("evidence_ref") not in evidence
                for item in model_visible
            ) or tuple(model_visible) not in ((), expected_visible):
                raise ValueError("environment checkpoint visibility mismatch")
            expected_budget = {
                "environment_actions": self._policy.max_actions,
                "media_operations": self._policy.media_operation_ceiling,
                "evidence_records": self._policy.max_evidence_records
                - len(evidence),
            }
            if (
                episode.environment_action_count != 0
                or episode.action_history
                or episode.remaining_budget != expected_budget
                or episode.revealed_assets != self._initial_visible
            ):
                raise ValueError("environment checkpoint counters mismatch")

            self._lifecycle = lifecycle
            self._revealed_assets = list(episode.revealed_assets)
            self._evidence = evidence
            self._model_visible_media = tuple(model_visible)
            self._last_visibility_disposition = payload.get(
                "last_visibility_disposition"
            )
            self._model_call_count = episode.model_call_count
            self._last_new_evidence = ()
            self._last_action_result = dict(payload["last_action_result"])
            safe_error = payload.get("last_safe_error")
            self._last_safe_error = None if safe_error is None else dict(safe_error)
            self._observation(emit_event=False)
        except Exception:
            raise OfflineCheckpointError("environment checkpoint validation failed") from None

        self._restored_from_checkpoint = True
        self._emit(
            "StateUpdated",
            {
                "state_revision": self._lifecycle.revision,
                "resume_restored": True,
                "evidence_refs": list(self._evidence),
                "model_call_count": self._model_call_count,
            },
        )

    def _validate_fixed_evidence(
        self,
        evidence: tuple[EvidenceRecord, ...],
    ) -> tuple[dict[str, Any], ...]:
        if not evidence or self._fixed_initial_media_records is None:
            return ()
        assert self._direct_presentation_asset_ids is not None
        expected: dict[str, EvidenceRecord] = {}
        visible: list[dict[str, Any]] = []
        for asset_id, record in zip(
            self._direct_presentation_asset_ids,
            self._fixed_initial_media_records,
            strict=True,
        ):
            binding = VisiblePublicEvidenceBinding.from_media(
                benchmark_id=self._task.benchmark_id,
                sample_id=self._task.sample_id,
                public_asset_id=asset_id,
                descriptor=self._assets[asset_id],
                record=record,
            )
            expected.setdefault(binding.evidence.evidence_id, binding.evidence)
            visible.append(binding.observation_projection())
        if evidence != tuple(expected.values()):
            raise ValueError("environment checkpoint fixed evidence mismatch")
        return tuple(visible)

    def _initial_observation(self) -> Observation:
        return Observation(
            episode_id=self._episode_id,
            state_revision=1,
            lifecycle_state=EpisodeLifecycleState.ACTIVE,
            task_summary=self._task,
            visible_assets=tuple(
                self._model_visible_asset_descriptor(self._assets[asset_id])
                for asset_id in self._initial_visible
            ),
            new_evidence=(),
            current_evidence_index=(),
            valid_submission_evidence_refs=(),
            model_visible_media=(),
            evidence_binding_contract_hash=EVIDENCE_BINDING_CONTRACT_HASH,
            action_result={"status": "reset", "revealed": list(self._initial_visible)},
            available_actions=self._policy.allowed_actions,
            remaining_budget={
                "environment_actions": self._policy.max_actions,
                "media_operations": self._policy.media_operation_ceiling,
                "evidence_records": self._policy.max_evidence_records,
            },
            terminal=False,
        )

    def _query_step(self, envelope: ActionEnvelope, semantic: str) -> Observation:
        if envelope.action is ActionType.GET_TASK_CONTEXT:
            result = {"task": self._task.to_dict()}
        elif envelope.action is ActionType.LIST_ASSETS:
            result = {"assets": [item.to_dict() for item in self._assets.values()]}
        elif envelope.action is ActionType.QUERY_STATE:
            result = {"state": self.episode_state.to_dict()}
        elif envelope.action is ActionType.LIST_EVIDENCE:
            result = {"evidence_refs": list(self._evidence)}
        else:
            raise OfflineArenaContractError("unsupported query action")
        return self._accept(envelope, semantic, result, new_evidence=())

    def _media_step(self, envelope: ActionEnvelope, semantic: str) -> Observation:
        if self._transport is None:
            raise OfflineArenaContractError("media transport is unavailable")
        args = dict(envelope.arguments)
        action = envelope.action
        source_asset_id: str | None = None
        compose_asset_ids: tuple[str, ...] = ()
        compose_sources: tuple[MediaTransportRecord, ...] = ()
        if (
            self._media_action_resolver is not None
            and action in {ActionType.GET_FRAME, ActionType.GET_FRAME_WINDOW}
        ):
            source_asset_id = str(args["asset_id"])
            resolved = self._media_action_resolver(envelope)
            if isinstance(resolved, MediaTransportRecord):
                result: MediaTransportRecord | Sequence[MediaTransportRecord] = resolved
            elif (
                isinstance(resolved, Sequence)
                and not isinstance(resolved, (str, bytes, bytearray))
                and resolved
                and all(isinstance(item, MediaTransportRecord) for item in resolved)
            ):
                result = resolved
            else:
                raise OfflineArenaContractError(
                    "adapter media resolver returned an invalid record"
                )
        elif action in {
            ActionType.OPEN_ASSET,
            ActionType.GET_VIEW,
            ActionType.GET_FRAME,
            ActionType.GET_FRAME_WINDOW,
        }:
            asset_id = str(args.pop("asset_id"))
            source_asset_id = asset_id
            source = self._record_or_source(asset_id)
            if action is ActionType.OPEN_ASSET:
                result = self._transport.open_asset(
                    source,
                    expected_sha256=self._expected_source_hash(asset_id),
                )
            elif action is ActionType.GET_VIEW:
                result = self._transport.get_view(
                    asset_id,
                    source=source,
                    expected_sha256=self._expected_source_hash(asset_id),
                )
            elif action is ActionType.GET_FRAME:
                result = self._transport.get_frame(source, **args)
            else:
                result = self._transport.get_frame_window(source, **args)
        elif action in {ActionType.CROP_REGION, ActionType.ZOOM_REGION}:
            asset_id = str(args.pop("asset_id"))
            source_asset_id = asset_id
            source = self._record_for_image(asset_id)
            result = (
                self._transport.crop_region(source, **args)
                if action is ActionType.CROP_REGION
                else self._transport.zoom_region(source, **args)
            )
        elif action is ActionType.COMPOSE_ASSETS:
            compose_asset_ids = tuple(args.pop("asset_ids"))
            compose_sources = tuple(
                self._record_for_image(asset_id)
                for asset_id in compose_asset_ids
            )
            result = self._transport.compose_assets(
                compose_sources,
                **args,
            )
        else:
            raise OfflineArenaContractError("unsupported media action")
        records = (
            list(result)
            if isinstance(result, Sequence)
            and not isinstance(result, (str, bytes, bytearray, MediaTransportRecord))
            else [result]
        )
        if not all(isinstance(record, MediaTransportRecord) for record in records):
            raise OfflineArenaContractError("media action returned an invalid record")
        root_groups = self._root_bindings_for_action(
            action,
            records,
            source_asset_id=source_asset_id,
            compose_asset_ids=compose_asset_ids,
            compose_sources=compose_sources,
        )
        for record, roots in zip(records, root_groups):
            self._register_media_record(
                record,
                roots=roots,
                operation=action.value,
            )
        self._media_action_count += 1
        action_result: dict[str, Any] = {
            "derived_asset_ids": [record.asset_id for record in records],
        }
        # Cache residency is runtime-local and can change across a deterministic
        # action replay. Keep it out of the visual-sufficiency transition while
        # preserving the legacy observation contract for every existing path.
        if not self._visual_sufficiency_binding:
            action_result["cache_hit"] = all(record.cache_hit for record in records)
        return self._accept(
            envelope,
            semantic,
            action_result,
            new_evidence=(),
        )

    def _record_evidence(
        self,
        envelope: ActionEnvelope,
        semantic: str,
    ) -> Observation:
        if len(self._evidence) >= self._policy.max_evidence_records:
            return self._reject_envelope(
                envelope, "evidence_record_budget_exhausted"
            )
        args = envelope.arguments
        asset_id = args.get("asset_id")
        if asset_id is not None and asset_id not in self._all_assets():
            raise OfflineArenaContractError("evidence references an unknown asset")
        media_hash = None
        if asset_id is not None:
            if asset_id in self._media_records:
                media_hash = self._media_records[asset_id].content_sha256
            else:
                media_hash = self._assets[asset_id].source_media_hash
        evidence_id = "evidence-" + hashlib.sha256(
            f"{self._episode_id}:{semantic}".encode("utf-8")
        ).hexdigest()[:32]
        record = EvidenceRecord(
            evidence_id=evidence_id,
            source_kind="model_estimated",
            modality=args["modality"],
            asset_id=asset_id,
            coordinate_frame=args.get("coordinate_frame", "unknown"),
            scale_type=args.get("scale_type", "unknown"),
            region=args.get("region"),
            frame=args.get("frame"),
            view=args.get("view"),
            producer_action_id=envelope.action_id,
            media_hash=media_hash,
            confidence=args.get("confidence"),
            provenance=args["provenance"],
        )
        self._evidence.setdefault(record.evidence_id, record)
        self._emit("EvidenceCreated", {"evidence_id": record.evidence_id})
        return self._accept(
            envelope,
            semantic,
            {"evidence_id": record.evidence_id},
            new_evidence=(record,),
        )

    def _submit_envelope(
        self,
        envelope: ActionEnvelope,
        semantic: str,
    ) -> Observation:
        try:
            submission = ParsedSubmission.from_envelope(envelope)
        except OfflineArenaContractError:
            return self._reject_envelope(envelope, "submission_contract_error")
        if (
            self._policy.submit_precondition
            == ONE_SUCCESSFUL_ANSWER_BEARING_MEDIA_OBSERVATION
            and not self._has_successful_answer_bearing_media_observation()
        ):
            return self._reject_submit_precondition(envelope, semantic)
        if self._policy.submit_precondition == MODALITY_SUFFICIENT_EVIDENCE:
            requirement = self._evidence_sufficiency_requirement
            if requirement is None:
                raise OfflineArenaContractError(
                    "typed sufficiency precondition lacks its requirement"
                )
            sufficiency = evaluate_evidence_sufficiency(
                requirement,
                tuple(self._observed_evidence_bundles),
            )
            if not sufficiency.satisfied:
                return self._reject_modality_sufficiency(
                    envelope,
                    semantic,
                    sufficiency.feedback,
                )
        invalid_refs = tuple(
            ref for ref in submission.evidence_refs if ref not in self._evidence
        )
        if invalid_refs:
            return self._reject_unknown_evidence_refs(envelope, invalid_refs)
        self._submission = submission
        self._register_action(envelope, semantic, "accepted", {"submitted": True})
        self._lifecycle.transition(EpisodeLifecycleState.SUBMITTED)
        self._last_new_evidence = ()
        self._last_safe_error = None
        self._last_action_result = {
            "status": "accepted",
            "submitted": True,
            "evidence_support_status": (
                "provided" if submission.evidence_refs else "not_provided"
            ),
            "evidence_count": len(submission.evidence_refs),
        }
        self._semantic_cache[semantic] = dict(self._last_action_result)
        if submission.evidence_refs and self._visual_sufficiency_binding:
            self._emit_evidence_lifecycle_stage(
                "consumed",
                submission.evidence_refs,
            )
        self._emit(
            "SubmissionReceived",
            {
                "evidence_count": len(submission.evidence_refs),
                "evidence_refs": list(submission.evidence_refs),
                "evidence_support_status": (
                    "provided" if submission.evidence_refs else "not_provided"
                ),
            },
        )
        self._emit("StateUpdated", {"state_revision": self._lifecycle.revision})
        self._commit()
        return self._observation(new_evidence=())

    def _reject_modality_sufficiency(
        self,
        envelope: ActionEnvelope,
        semantic: str,
        feedback: Any,
    ) -> Observation:
        if feedback is None:
            raise OfflineArenaContractError(
                "insufficient typed evidence must provide correction feedback"
            )
        typed_feedback = {
            "requirement_id": feedback.requirement_id,
            "modality": feedback.modality,
            "required_evidence": dict(feedback.required_evidence),
            "current_evidence": dict(feedback.current_evidence),
            "missing_evidence": list(feedback.missing_evidence),
            "allowed_actions": list(feedback.allowed_actions),
            "empty_reference_disclosure": False,
            "retryable_with_new_model_turn": True,
        }
        self._register_action(
            envelope,
            semantic,
            "rejected",
            typed_feedback,
        )
        self._last_new_evidence = ()
        self._last_safe_error = dict(typed_feedback)
        self._last_action_result = {
            "status": "rejected",
            "executed": False,
            "semantic_progress": False,
        }
        self._emit(
            "ActionRejected",
            {
                "action": envelope.action.value,
                **typed_feedback,
                "state_revision_advanced": False,
            },
        )
        self._commit()
        return self._observation(new_evidence=())

    def _reject_submit_precondition(
        self,
        envelope: ActionEnvelope,
        semantic: str,
    ) -> Observation:
        feedback = {
            "error_type": MEDIA_OBSERVATION_REQUIRED_BEFORE_SUBMIT,
            "submit_precondition": self._policy.submit_precondition,
            "media_actions": [
                action.value
                for action in self._policy.allowed_actions
                if action in _MEDIA_ACTIONS
            ],
            "satisfied": False,
            "retryable_with_new_model_turn": True,
        }
        self._register_action(envelope, semantic, "rejected", feedback)
        self._last_new_evidence = ()
        self._last_safe_error = dict(feedback)
        self._last_action_result = {
            "status": "rejected",
            "executed": False,
            "semantic_progress": False,
        }
        self._emit(
            "ActionRejected",
            {
                "action": envelope.action.value,
                **feedback,
                "state_revision_advanced": False,
            },
        )
        self._commit()
        return self._observation(new_evidence=())

    def _accept(
        self,
        envelope: ActionEnvelope,
        semantic: str,
        result: Mapping[str, Any],
        *,
        new_evidence: tuple[EvidenceRecord, ...],
    ) -> Observation:
        self._register_action(envelope, semantic, "accepted", result)
        self._lifecycle.touch()
        self._last_new_evidence = new_evidence
        self._last_safe_error = None
        self._last_action_result = {"status": "accepted", **dict(result)}
        self._semantic_cache[semantic] = dict(self._last_action_result)
        output_count = 0
        derived_asset_ids = result.get("derived_asset_ids")
        if isinstance(derived_asset_ids, list):
            output_count = len(derived_asset_ids)
        self._emit(
            "ActionExecuted",
            {
                "action": envelope.action.value,
                "action_id": envelope.action_id,
                "output_count": output_count,
            },
        )
        self._emit("StateUpdated", {"state_revision": self._lifecycle.revision})
        self._commit()
        return self._observation(new_evidence=new_evidence)

    def _record_cached(
        self,
        envelope: ActionEnvelope,
        semantic: str,
        cached: Mapping[str, Any],
    ) -> Observation:
        self._register_action(envelope, semantic, "cache_hit", cached)
        self._lifecycle.touch()
        self._last_new_evidence = ()
        self._last_safe_error = None
        self._last_action_result = {
            **dict(cached),
            "status": "cache_hit",
            "cache_hit": True,
        }
        self._emit(
            "ActionExecuted",
            {
                "action": envelope.action.value,
                "action_id": envelope.action_id,
                "cache_hit": True,
                "output_count": len(cached.get("derived_asset_ids", []))
                if isinstance(cached.get("derived_asset_ids"), list)
                else 0,
            },
        )
        self._emit("StateUpdated", {"state_revision": self._lifecycle.revision})
        self._commit()
        return self._observation(new_evidence=())

    def _reject_unparsed(self, error_type: str) -> Observation:
        if self._environment_action_count >= self._policy.max_actions:
            return self._block("environment_action_budget_exhausted")
        self._environment_action_count += 1
        self._lifecycle.touch()
        self._last_new_evidence = ()
        self._last_safe_error = {
            "error_type": error_type,
            "message": "action contract rejected",
            "retryable_with_new_model_turn": True,
        }
        self._last_action_result = {"status": "rejected", "executed": False}
        self._action_history.append(
            {"status": "rejected", "error_type": error_type, "envelope": None}
        )
        self._emit("ActionRejected", {"error_type": error_type})
        self._emit("StateUpdated", {"state_revision": self._lifecycle.revision})
        self._commit()
        return self._observation(new_evidence=())

    def _reject_envelope(
        self,
        envelope: ActionEnvelope,
        error_type: str,
    ) -> Observation:
        semantic = canonical_sha256(envelope.to_dict())
        self._register_action(envelope, semantic, "rejected", {})
        self._lifecycle.touch()
        self._last_new_evidence = ()
        self._last_safe_error = {
            "error_type": error_type,
            "message": "action could not be executed",
            "retryable_with_new_model_turn": True,
        }
        self._last_action_result = {"status": "rejected", "executed": False}
        self._emit(
            "ActionRejected",
            {"action": envelope.action.value, "error_type": error_type},
        )
        self._emit("StateUpdated", {"state_revision": self._lifecycle.revision})
        self._commit()
        return self._observation(new_evidence=())

    def _reject_unknown_evidence_refs(
        self,
        envelope: ActionEnvelope,
        invalid_refs: tuple[str, ...],
    ) -> Observation:
        semantic = canonical_sha256(envelope.to_dict())
        self._register_action(envelope, semantic, "rejected", {})
        self._lifecycle.touch()
        self._last_new_evidence = ()
        self._last_safe_error = {
            "error_type": "UNKNOWN_EVIDENCE_REF",
            "invalid_refs": list(invalid_refs),
            "valid_submission_evidence_refs": list(self._evidence),
            "empty_list_allowed": True,
            "retryable_with_new_model_turn": True,
        }
        self._last_action_result = {"status": "rejected", "executed": False}
        self._emit(
            "ActionRejected",
            {
                "action": envelope.action.value,
                "error_type": "UNKNOWN_EVIDENCE_REF",
                "invalid_ref_count": len(invalid_refs),
                "invalid_refs": list(invalid_refs),
                "valid_submission_evidence_refs": list(self._evidence),
                "empty_list_allowed": True,
                "retryable_with_new_model_turn": True,
            },
        )
        self._emit("StateUpdated", {"state_revision": self._lifecycle.revision})
        self._commit()
        return self._observation(new_evidence=())

    def _block(self, reason: str) -> Observation:
        self._lifecycle.transition(
            EpisodeLifecycleState.BLOCKED,
            failure_code=reason,
        )
        self._last_safe_error = {
            "error_type": "budget_exhaustion",
            "message": reason,
            "retryable_with_new_model_turn": False,
        }
        self._last_action_result = {"status": "blocked", "reason": reason}
        self._emit(
            "EpisodeTerminated",
            {"status": "BLOCKED", "reason": reason},
        )
        self._emit("StateUpdated", {"state_revision": self._lifecycle.revision})
        self._commit()
        return self._observation(new_evidence=())

    def _register_action(
        self,
        envelope: ActionEnvelope,
        semantic: str,
        status: str,
        result: Mapping[str, Any],
    ) -> None:
        self._environment_action_count += 1
        self._action_history.append(
            {
                "action_id": envelope.action_id,
                "semantic_hash": semantic,
                "family": envelope.action.family,
                "envelope": envelope.to_dict(),
                "status": status,
                "result": json_copy(result, "action result", public_task=True),
            }
        )

    def _root_bindings_for_action(
        self,
        action: ActionType,
        records: Sequence[MediaTransportRecord],
        *,
        source_asset_id: str | None,
        compose_asset_ids: Sequence[str],
        compose_sources: Sequence[MediaTransportRecord],
    ) -> tuple[tuple[dict[str, Any], ...], ...]:
        requirement = self._evidence_sufficiency_requirement
        empty = tuple(() for _ in records)
        if requirement is None:
            return empty
        if action in {ActionType.GET_FRAME, ActionType.GET_FRAME_WINDOW}:
            if requirement.modality is not VisualModality.VIDEO:
                return empty
            groups: list[tuple[dict[str, Any], ...]] = []
            for record in records:
                frame_index = record.frame_index
                if (
                    source_asset_id is None
                    or isinstance(frame_index, bool)
                    or not isinstance(frame_index, int)
                ):
                    groups.append(())
                    continue
                root_digest = hashlib.sha256(
                    f"{source_asset_id}:{frame_index}".encode("utf-8")
                ).hexdigest()
                groups.append(
                    (
                        {
                            "asset_id": f"frame-root-{root_digest[:32]}",
                            "media_hash": record.content_sha256,
                            "frame_index": frame_index,
                        },
                    )
                )
            return tuple(groups)
        if action in {
            ActionType.OPEN_ASSET,
            ActionType.GET_VIEW,
            ActionType.CROP_REGION,
            ActionType.ZOOM_REGION,
        }:
            roots = (
                ()
                if source_asset_id is None
                else self._root_bindings_for_asset(source_asset_id)
            )
            return tuple(tuple(dict(root) for root in roots) for _ in records)
        if action is not ActionType.COMPOSE_ASSETS:
            return empty
        if requirement.modality is not VisualModality.MULTI_VIEW:
            return empty
        expected_parents = tuple(record.asset_id for record in compose_sources)
        input_roots = tuple(
            self._root_bindings_for_asset(asset_id)
            for asset_id in compose_asset_ids
        )
        if not input_roots or any(not roots for roots in input_roots):
            return empty
        flattened: list[dict[str, Any]] = []
        seen: set[tuple[str, str, int | None]] = set()
        for roots in input_roots:
            for root in roots:
                key = self._root_key(root)
                if key not in seen:
                    flattened.append(dict(root))
                    seen.add(key)
        return tuple(
            tuple(flattened)
            if tuple(record.parent_asset_ids) == expected_parents
            else ()
            for record in records
        )

    def _root_bindings_for_asset(
        self,
        asset_id: str,
    ) -> tuple[dict[str, Any], ...]:
        inherited = self._media_root_bindings.get(asset_id)
        if inherited is not None:
            return tuple(dict(root) for root in inherited)
        requirement = self._evidence_sufficiency_requirement
        descriptor = self._assets.get(asset_id)
        if requirement is None or descriptor is None:
            return ()
        if requirement.modality is VisualModality.VIDEO:
            return ()
        return (
            {
                "asset_id": descriptor.asset_id,
                "media_hash": descriptor.source_media_hash,
                "frame_index": None,
            },
        )

    def _media_roots_for_visibility(
        self,
        public_asset_id: str,
        _descriptor: AssetDescriptor,
        record: MediaTransportRecord,
    ) -> tuple[dict[str, Any], ...]:
        roots = self._media_root_bindings.get(record.asset_id)
        if roots is None:
            roots = self._media_root_bindings.get(public_asset_id)
        if roots is None:
            roots = self._root_bindings_for_asset(public_asset_id)
        return tuple(dict(root) for root in roots)

    @staticmethod
    def _root_key(root: Mapping[str, Any]) -> tuple[str, str, int | None]:
        return (
            str(root["asset_id"]),
            str(root["media_hash"]),
            root.get("frame_index"),
        )

    def _record_observed_root_bundles(self) -> None:
        requirement = self._evidence_sufficiency_requirement
        if requirement is None:
            return
        for presentation in self._pending_root_presentations:
            operation = presentation.get("operation")
            if operation not in requirement.allowed_actions:
                continue
            raw_roots = presentation.get("roots")
            if not isinstance(raw_roots, list):
                continue
            roots = tuple(
                root
                for root in raw_roots
                if isinstance(root, Mapping)
                and self._root_key(root) not in self._observed_root_keys
            )
            if not roots:
                continue
            asset_ids = tuple(str(root["asset_id"]) for root in roots)
            media_hashes = tuple(str(root["media_hash"]) for root in roots)
            frame_indices = (
                tuple(int(root["frame_index"]) for root in roots)
                if requirement.modality is VisualModality.VIDEO
                else ()
            )
            if requirement.modality is VisualModality.VIDEO:
                ordered = all(
                    left < right
                    for left, right in zip(frame_indices, frame_indices[1:])
                )
                if (
                    len(frame_indices) > 1
                    and requirement.frame_count is not None
                    and requirement.frame_count > 1
                ):
                    temporal_span_ratio = (
                        max(frame_indices) - min(frame_indices)
                    ) / (requirement.frame_count - 1)
                else:
                    temporal_span_ratio = 0.0
                distinct_count = min(
                    len(set(frame_indices)),
                    len(set(media_hashes)),
                )
            else:
                ordered = True
                temporal_span_ratio = None
                distinct_count = min(
                    len(set(asset_ids)),
                    len(set(media_hashes)),
                )
            evidence_ref = str(presentation["evidence_ref"])
            bundle = ModalityEvidenceBundle(
                modality=requirement.modality,
                source_asset_ids=asset_ids,
                source_media_hashes=media_hashes,
                operation=str(operation),
                ordered=ordered,
                distinct_count=distinct_count,
                presented_count=len(roots),
                temporal_span_ratio=temporal_span_ratio,
                frame_indices=frame_indices,
                sequence_positions=tuple(range(len(roots))),
                evidence_refs=(evidence_ref,) * len(roots),
                payload_bytes=int(presentation["payload_bytes"]),
                private_provenance={
                    "root_identity_policy": "public_source_root_v1",
                },
            )
            self._observed_evidence_bundles.append(bundle)
            self._observed_root_keys.update(self._root_key(root) for root in roots)

    def _register_media_record(
        self,
        record: MediaTransportRecord,
        *,
        roots: Sequence[Mapping[str, Any]] = (),
        operation: str | None = None,
    ) -> None:
        """Register a derived asset without claiming the model has seen it."""

        self._media_records[record.asset_id] = record
        self._media_root_bindings[record.asset_id] = tuple(
            dict(root) for root in roots
        )
        self._media_root_operations[record.asset_id] = operation or record.operation
        if record.asset_id not in self._revealed_assets:
            self._revealed_assets.append(record.asset_id)
        descriptor = AssetDescriptor(
            asset_id=record.asset_id,
            modality="image" if record.media_kind != "video" else "video",
            source_kind="derived" if record.parent_asset_ids else "public_raw",
            width=record.width,
            height=record.height,
            frame_count=record.frame_count,
            duration=(
                None
                if record.duration_ms is None
                else record.duration_ms / 1000.0
            ),
            public=True,
            source_media_hash=record.content_sha256,
            metadata={
                "operation": record.operation,
                "view_id": record.view_id,
                "frame_id": record.frame_id,
                "frame_index": record.frame_index,
                "timestamp_ms": record.timestamp_ms,
            },
        )
        self._derived_assets[record.asset_id] = descriptor

    def _record_or_source(self, asset_id: str) -> Any:
        if asset_id in self._media_records:
            return self._media_records[asset_id]
        if asset_id in self._asset_sources:
            return self._asset_sources[asset_id]
        raise OfflineArenaContractError("asset source is unavailable")

    def _record_for_image(self, asset_id: str) -> MediaTransportRecord:
        value = self._record_or_source(asset_id)
        if isinstance(value, MediaTransportRecord):
            return value
        if self._transport is None:
            raise OfflineArenaContractError("media transport is unavailable")
        return self._transport.open_asset(
            value,
            expected_sha256=self._expected_source_hash(asset_id),
        )

    def _expected_source_hash(self, asset_id: str) -> str | None:
        if asset_id in self._source_content_sha256_overrides:
            return self._source_content_sha256_overrides[asset_id]
        descriptor = self._assets.get(asset_id)
        return None if descriptor is None else descriptor.source_media_hash

    def _has_successful_answer_bearing_media_observation(self) -> bool:
        return any(
            record.source_kind in {"public_raw", "cpu_derived"}
            for record in self._evidence.values()
        )

    def _visual_video_binding(self) -> bool:
        if self._modality_evidence_bundle is not None:
            return self._modality_evidence_bundle.modality is VisualModality.VIDEO
        if self._evidence_sufficiency_requirement is not None:
            return (
                self._evidence_sufficiency_requirement.modality
                is VisualModality.VIDEO
            )
        return False

    def _model_visible_asset_descriptor(
        self,
        descriptor: AssetDescriptor,
    ) -> AssetDescriptor:
        if not self._visual_sufficiency_binding:
            return descriptor
        metadata = {
            key: value
            for key, value in descriptor.metadata.items()
            if not any(
                part in str(key).casefold().replace("-", "_")
                for part in _PRIVATE_TEMPORAL_METADATA_PARTS
            )
        }
        if metadata == descriptor.metadata:
            return descriptor
        return replace(descriptor, metadata=metadata)

    def _emit_evidence_lifecycle_stage(
        self,
        stage: str,
        evidence_refs: Sequence[str],
        *,
        request_id: str | None = None,
    ) -> None:
        refs = list(dict.fromkeys(evidence_refs))
        if not refs:
            return
        payload: dict[str, Any] = {
            "state_revision": self._lifecycle.revision,
            "evidence_lifecycle_stage": stage,
            "evidence_refs": refs,
        }
        if request_id is not None:
            payload["request_id"] = request_id
        self._emit("StateUpdated", payload)

    def _all_assets(self) -> dict[str, AssetDescriptor]:
        return {**self._assets, **self._derived_assets}

    def _remaining_budget(self) -> dict[str, int]:
        return {
            "environment_actions": max(
                0,
                self._policy.max_actions - self._environment_action_count,
            ),
            "media_operations": max(
                0,
                self._policy.media_operation_ceiling - self._media_action_count,
            ),
            "evidence_records": max(
                0,
                self._policy.max_evidence_records - len(self._evidence),
            ),
        }

    def _submission_status(self) -> str:
        if self.state is EpisodeLifecycleState.EVALUATED:
            return "evaluated"
        if self.state is EpisodeLifecycleState.SUBMITTED:
            return "submitted"
        if self.state is EpisodeLifecycleState.BLOCKED:
            return "blocked"
        if self.state is EpisodeLifecycleState.CANCELED:
            return "canceled"
        if self.state is EpisodeLifecycleState.FAILED:
            return "failed"
        return "not_submitted"

    def _observation(
        self,
        *,
        new_evidence: tuple[EvidenceRecord, ...] | None = None,
        emit_event: bool = True,
    ) -> Observation:
        visible = self._all_assets()
        visible_assets = tuple(
            self._model_visible_asset_descriptor(visible[asset_id])
            for asset_id in self._revealed_assets
            if asset_id in visible
        )
        evidence = self._last_new_evidence if new_evidence is None else new_evidence
        available = (
            self._policy.allowed_actions
            if self.state is EpisodeLifecycleState.ACTIVE
            else ()
        )
        observation = Observation(
            episode_id=self._episode_id,
            state_revision=self._lifecycle.revision,
            lifecycle_state=self.state,
            task_summary=self._task,
            visible_assets=visible_assets,
            new_evidence=evidence,
            current_evidence_index=tuple(self._evidence),
            valid_submission_evidence_refs=tuple(self._evidence),
            model_visible_media=self._model_visible_media,
            evidence_binding_contract_hash=EVIDENCE_BINDING_CONTRACT_HASH,
            action_result=self._last_action_result,
            available_actions=available,
            remaining_budget=self._remaining_budget(),
            terminal=self.state.terminal,
            safe_error=self._last_safe_error,
        )
        if emit_event:
            self._emit(
                "ObservationEmitted",
                {"state_revision": self._lifecycle.revision},
            )
        return observation

    @staticmethod
    def _coerce_action(
        action: ActionEnvelope | Mapping[str, Any] | str,
    ) -> ActionEnvelope:
        if isinstance(action, ActionEnvelope):
            return action
        if isinstance(action, str):
            return ActionEnvelope.from_json(action)
        if isinstance(action, Mapping):
            return ActionEnvelope.from_mapping(action)
        raise OfflineArenaContractError("action must be a strict envelope")

    def _require_state(self, *states: EpisodeLifecycleState) -> None:
        try:
            self._lifecycle.require(*states)
        except StateTransitionError as exc:
            raise OfflineArenaLifecycleError(str(exc)) from None

    def _emit(self, event_type: str, payload: Mapping[str, Any]) -> None:
        event = {
            "event_type": event_type,
            "episode_id": self._episode_id,
            "state_revision": self._lifecycle.revision,
            "payload": json_copy(payload, "event payload", public_task=True),
        }
        self._events.append(event)
        if self._event_sink is not None:
            self._event_sink(event)

    def _commit(self) -> None:
        if self._checkpoint_store is None:
            return
        try:
            self._checkpoint_store.save(self.snapshot())
        except AtomicCheckpointError:
            self._lifecycle.checkpoint_failed()
            raise OfflineCheckpointError("atomic checkpoint commit failed") from None


OfflineArenaEnvironment = OfflineEvidenceEnvironment
OfflineEnvironment = OfflineEvidenceEnvironment


__all__ = [
    "ActionRejectedError",
    "EvaluationFailedError",
    "OfflineArenaEnvironment",
    "OfflineArenaError",
    "OfflineArenaLifecycleError",
    "OfflineCheckpointError",
    "OfflineEnvironment",
    "OfflineEvidenceEnvironment",
    "SubmissionRejectedError",
]
