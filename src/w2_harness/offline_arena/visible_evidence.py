"""Stable binding between model-visible public media and evidence handles."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from .contracts import AssetDescriptor, EvidenceRecord, OfflineArenaContractError
from .media_transport import MediaTransportRecord


EVIDENCE_BINDING_CONTRACT_ID = "visible_public_evidence_binding"
EVIDENCE_ID_POLICY = {
    "contract_id": EVIDENCE_BINDING_CONTRACT_ID,
    "identity_fields": [
        "benchmark_id",
        "sample_id",
        "canonical_asset_id",
        "source_kind",
        "content_sha256",
        "canonical_derivation_sha256",
        "view_id",
        "canonical_frame_id",
        "frame_index",
        "region",
    ],
    "excluded_fields": [
        "git_commit",
        "branch",
        "tree",
        "run_root",
        "episode_id",
        "profile",
        "action_id",
        "model_response",
        "task_answer",
    ],
    "digest": "sha256",
    "collision_policy": "fail_closed_on_divergent_record",
}


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


EVIDENCE_ID_POLICY_HASH = _canonical_sha256(EVIDENCE_ID_POLICY)
EVIDENCE_BINDING_CONTRACT = {
    "contract_id": EVIDENCE_BINDING_CONTRACT_ID,
    "evidence_id_policy_hash": EVIDENCE_ID_POLICY_HASH,
    "asset_ref_is_evidence_ref": False,
    "only_transported_media_is_bindable": True,
    "empty_submission_refs_allowed": True,
    "unknown_ref_policy": "fail_closed_with_exact_valid_ref_feedback",
    "visibility_commit_boundary": "terminal_provider_response_or_cache_hit",
}
EVIDENCE_BINDING_CONTRACT_HASH = _canonical_sha256(EVIDENCE_BINDING_CONTRACT)


def _source_kind(
    descriptor: AssetDescriptor,
    record: MediaTransportRecord,
) -> str:
    if (
        record.operation in {"OPEN_ASSET", "GET_VIEW"}
        and descriptor.source_kind not in {"derived", "preview", "contact_sheet"}
        and not record.parent_asset_ids
    ):
        return "public_raw"
    return "cpu_derived"


def _canonical_media_identity(
    public_asset_id: str,
    record: MediaTransportRecord,
) -> dict[str, Any]:
    is_frame = record.frame_index is not None or record.frame_id is not None
    if not is_frame:
        return {
            "asset_id": public_asset_id,
            "derivation_sha256": record.derivation_sha256,
            "frame_id": None,
            "operation": record.operation,
            "parent_asset_ids": list(record.parent_asset_ids),
            "transport_asset_id": record.asset_id,
        }
    canonical_asset_id = (
        record.parent_asset_ids[0]
        if len(record.parent_asset_ids) == 1
        else public_asset_id
    )
    derivation = _canonical_sha256(
        {
            "kind": "frame_access",
            "asset_id": canonical_asset_id,
            "content_sha256": record.content_sha256,
            "frame_index": record.frame_index,
            "timestamp_ms": record.timestamp_ms,
            "view_id": record.view_id,
        }
    )
    return {
        "asset_id": canonical_asset_id,
        "derivation_sha256": derivation,
        "frame_id": f"frame-{derivation[:32]}",
        "operation": "FRAME_ACCESS",
        "parent_asset_ids": [canonical_asset_id],
        "transport_asset_id": None,
    }


@dataclass(frozen=True)
class VisiblePublicEvidenceBinding:
    """One public media payload and the exact handle the model may submit."""

    public_asset_id: str
    transport_asset_id: str
    evidence: EvidenceRecord
    content_sha256: str
    derivation_sha256: str

    @classmethod
    def from_media(
        cls,
        *,
        benchmark_id: str,
        sample_id: str,
        public_asset_id: str,
        descriptor: AssetDescriptor,
        record: MediaTransportRecord,
        producer_action_id: str | None = None,
    ) -> "VisiblePublicEvidenceBinding":
        source_kind = _source_kind(descriptor, record)
        canonical = _canonical_media_identity(public_asset_id, record)
        frame = (
            None
            if record.frame_index is None and record.frame_id is None
            else {
                "frame_id": canonical["frame_id"],
                "frame_index": record.frame_index,
                "timestamp_ms": record.timestamp_ms,
            }
        )
        identity = {
            "benchmark_id": benchmark_id,
            "sample_id": sample_id,
            "canonical_asset_id": canonical["asset_id"],
            "source_kind": source_kind,
            "content_sha256": record.content_sha256,
            "canonical_derivation_sha256": canonical["derivation_sha256"],
            "view_id": record.view_id,
            "canonical_frame_id": canonical["frame_id"],
            "frame_index": record.frame_index,
            "region": None,
        }
        digest = _canonical_sha256(identity)
        prefix = "ev-public" if source_kind == "public_raw" else "ev-derived"
        evidence_id = f"{prefix}-{digest}"
        evidence = EvidenceRecord(
            evidence_id=evidence_id,
            source_kind=source_kind,
            modality="video" if record.media_kind == "video" else "image",
            asset_id=canonical["asset_id"],
            coordinate_frame="image",
            scale_type="unknown",
            region=None,
            frame=frame,
            view=record.view_id,
            producer_action_id=(producer_action_id or f"visibility-{digest[:32]}"),
            media_hash=record.content_sha256,
            confidence=None,
            provenance={
                "binding_contract_hash": EVIDENCE_BINDING_CONTRACT_HASH,
                "derivation_sha256": canonical["derivation_sha256"],
                "operation": canonical["operation"],
                "parent_asset_ids": canonical["parent_asset_ids"],
                "transport_asset_id": canonical["transport_asset_id"],
            },
        )
        return cls(
            public_asset_id=public_asset_id,
            transport_asset_id=record.asset_id,
            evidence=evidence,
            content_sha256=record.content_sha256,
            derivation_sha256=canonical["derivation_sha256"],
        )

    def media_fields(self) -> dict[str, Any]:
        return {
            "public_asset_id": self.public_asset_id,
            "evidence_ref": self.evidence.evidence_id,
            "evidence_source_kind": self.evidence.source_kind,
            "evidence_binding_contract_hash": EVIDENCE_BINDING_CONTRACT_HASH,
        }

    def observation_projection(self) -> dict[str, Any]:
        return {
            "asset_id": self.public_asset_id,
            "evidence_ref": self.evidence.evidence_id,
            "modality": self.evidence.modality,
            "view_id": self.evidence.view,
            "frame_id": (
                None
                if not isinstance(self.evidence.frame, Mapping)
                else self.evidence.frame.get("frame_id")
            ),
            "source_kind": self.evidence.source_kind,
        }


def assert_collision_safe(
    existing: EvidenceRecord | None,
    candidate: EvidenceRecord,
) -> None:
    if existing is not None and existing.to_dict() != candidate.to_dict():
        raise OfflineArenaContractError(
            "evidence identity collision has divergent canonical records"
        )


__all__ = [
    "EVIDENCE_BINDING_CONTRACT",
    "EVIDENCE_BINDING_CONTRACT_HASH",
    "EVIDENCE_BINDING_CONTRACT_ID",
    "EVIDENCE_ID_POLICY",
    "EVIDENCE_ID_POLICY_HASH",
    "VisiblePublicEvidenceBinding",
    "assert_collision_safe",
]
