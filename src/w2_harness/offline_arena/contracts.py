"""Typed, public-only contracts for deterministic offline evidence episodes."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .state import EpisodeLifecycleState


STATE_SCHEMA_VERSION = "w2-offline-arena-state-v1.0"
ONE_SUCCESSFUL_ANSWER_BEARING_MEDIA_OBSERVATION = (
    "one_successful_answer_bearing_media_observation"
)
MODALITY_SUFFICIENT_EVIDENCE = "modality_sufficient_evidence"
MEDIA_OBSERVATION_REQUIRED_BEFORE_SUBMIT = (
    "MEDIA_OBSERVATION_REQUIRED_BEFORE_SUBMIT"
)
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+-]{0,191}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PRIVATE_TASK_KEYS = {
    "answer",
    "correct_answer",
    "expected_answer",
    "ground_truth",
    "groundtruth",
    "hidden_label",
    "private_reference",
    "private_truth",
    "reference_answer",
    "target_answer",
}
_SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "password",
    "secret",
    "token",
)


class OfflineArenaContractError(ValueError):
    """A value cannot cross the offline arena's public contract boundary."""


def _identifier(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value.strip()):
        raise OfflineArenaContractError(
            f"{field_name} must be a portable non-empty identifier"
        )
    return value.strip()


def _text(value: Any, field_name: str, *, maximum: int = 8192) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OfflineArenaContractError(f"{field_name} must be non-empty text")
    normalized = value.strip()
    if len(normalized) > maximum:
        raise OfflineArenaContractError(
            f"{field_name} must be at most {maximum} characters"
        )
    return normalized


def _normalized_key(value: Any) -> str:
    return str(value).casefold().replace("-", "_").replace(" ", "_")


def _contains_private_task_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _normalized_key(key) in _PRIVATE_TASK_KEYS:
                return True
            if _contains_private_task_key(item):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_private_task_key(item) for item in value)
    return False


def _contains_sensitive_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = _normalized_key(key)
            if any(part in normalized for part in _SENSITIVE_KEY_PARTS):
                return True
            if _contains_sensitive_key(item):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_sensitive_key(item) for item in value)
    return False


def json_copy(
    value: Any,
    field_name: str,
    *,
    public_task: bool = False,
) -> Any:
    """Return an isolated finite-JSON copy, rejecting private public-task keys."""

    if public_task and _contains_private_task_key(value):
        raise OfflineArenaContractError(
            f"{field_name} contains a private reference field"
        )
    if _contains_sensitive_key(value):
        raise OfflineArenaContractError(
            f"{field_name} contains a credential-bearing field"
        )
    try:
        serialized = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
        )
        return copy.deepcopy(json.loads(serialized))
    except (TypeError, ValueError) as exc:
        raise OfflineArenaContractError(
            f"{field_name} must contain finite JSON values"
        ) from exc


def _string_tuple(
    value: Any,
    field_name: str,
    *,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value, Sequence
    ):
        raise OfflineArenaContractError(f"{field_name} must be a sequence")
    result = tuple(_identifier(item, field_name) for item in value)
    if not allow_empty and not result:
        raise OfflineArenaContractError(f"{field_name} must not be empty")
    if len(set(result)) != len(result):
        raise OfflineArenaContractError(f"{field_name} must be unique")
    return result


@dataclass(frozen=True)
class AssetDescriptor:
    """One opaque public asset descriptor with no local filesystem handle."""

    asset_id: str
    modality: str
    source_kind: str
    width: int | None
    height: int | None
    frame_count: int | None
    duration: float | None
    public: bool
    source_media_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "asset_id", _identifier(self.asset_id, "asset_id"))
        object.__setattr__(
            self, "modality", _identifier(self.modality, "asset.modality")
        )
        object.__setattr__(
            self, "source_kind", _identifier(self.source_kind, "asset.source_kind")
        )
        for name in ("width", "height", "frame_count"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise OfflineArenaContractError(f"asset.{name} must be positive or null")
        if self.duration is not None and (
            isinstance(self.duration, bool)
            or not isinstance(self.duration, (int, float))
            or not math.isfinite(float(self.duration))
            or float(self.duration) < 0.0
        ):
            raise OfflineArenaContractError("asset.duration must be non-negative or null")
        if self.duration is not None:
            object.__setattr__(self, "duration", float(self.duration))
        if self.public is not True:
            raise OfflineArenaContractError("model-visible AssetDescriptor must be public")
        if not isinstance(self.source_media_hash, str) or not _SHA256.fullmatch(
            self.source_media_hash
        ):
            raise OfflineArenaContractError(
                "asset.source_media_hash must be a lowercase SHA-256"
            )
        if not isinstance(self.metadata, Mapping):
            raise OfflineArenaContractError("asset.metadata must be an object")
        object.__setattr__(
            self,
            "metadata",
            json_copy(self.metadata, "asset.metadata", public_task=True),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "modality": self.modality,
            "source_kind": self.source_kind,
            "width": self.width,
            "height": self.height,
            "frame_count": self.frame_count,
            "duration": self.duration,
            "public": self.public,
            "source_media_hash": self.source_media_hash,
            "metadata": json_copy(self.metadata, "asset.metadata"),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AssetDescriptor":
        required = {
            "asset_id",
            "modality",
            "source_kind",
            "width",
            "height",
            "frame_count",
            "duration",
            "public",
            "source_media_hash",
        }
        optional = {"metadata"}
        if not isinstance(value, Mapping):
            raise OfflineArenaContractError("asset must be an object")
        if not required <= set(value) or set(value) - required - optional:
            raise OfflineArenaContractError("asset fields are not canonical")
        return cls(
            asset_id=value["asset_id"],
            modality=value["modality"],
            source_kind=value["source_kind"],
            width=value["width"],
            height=value["height"],
            frame_count=value["frame_count"],
            duration=value["duration"],
            public=value["public"],
            source_media_hash=value["source_media_hash"],
            metadata=value.get("metadata", {}),
        )


@dataclass(frozen=True)
class PublicTask:
    """The complete model-visible task, structurally unable to carry truth."""

    benchmark_id: str
    sample_id: str
    question: str
    choices: tuple[Any, ...]
    answer_format: str
    public_metadata: dict[str, Any]
    asset_ids: tuple[str, ...]
    observation_policy: str
    category: str
    submission_schema_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "benchmark_id", _identifier(self.benchmark_id, "benchmark_id")
        )
        object.__setattr__(self, "sample_id", _identifier(self.sample_id, "sample_id"))
        object.__setattr__(self, "question", _text(self.question, "question"))
        if isinstance(self.choices, (str, bytes, bytearray)) or not isinstance(
            self.choices, Sequence
        ):
            raise OfflineArenaContractError("choices must be a sequence")
        choices = tuple(
            json_copy(item, "choice", public_task=True) for item in self.choices
        )
        serialized_choices = [
            json.dumps(item, ensure_ascii=True, sort_keys=True) for item in choices
        ]
        if len(set(serialized_choices)) != len(serialized_choices):
            raise OfflineArenaContractError("choices must be unique")
        object.__setattr__(self, "choices", choices)
        object.__setattr__(
            self, "answer_format", _identifier(self.answer_format, "answer_format")
        )
        if not isinstance(self.public_metadata, Mapping):
            raise OfflineArenaContractError("public_metadata must be an object")
        object.__setattr__(
            self,
            "public_metadata",
            json_copy(self.public_metadata, "public_metadata", public_task=True),
        )
        object.__setattr__(
            self, "asset_ids", _string_tuple(self.asset_ids, "asset_ids")
        )
        object.__setattr__(
            self,
            "observation_policy",
            _identifier(self.observation_policy, "observation_policy"),
        )
        object.__setattr__(self, "category", _text(self.category, "category", maximum=512))
        object.__setattr__(
            self,
            "submission_schema_id",
            _identifier(self.submission_schema_id, "submission_schema_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "sample_id": self.sample_id,
            "question": self.question,
            "choices": json_copy(list(self.choices), "choices"),
            "answer_format": self.answer_format,
            "public_metadata": json_copy(self.public_metadata, "public_metadata"),
            "asset_ids": list(self.asset_ids),
            "observation_policy": self.observation_policy,
            "category": self.category,
            "submission_schema_id": self.submission_schema_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PublicTask":
        required = {
            "benchmark_id",
            "sample_id",
            "question",
            "choices",
            "answer_format",
            "public_metadata",
            "asset_ids",
            "observation_policy",
            "category",
            "submission_schema_id",
        }
        if not isinstance(value, Mapping):
            raise OfflineArenaContractError("public task must be an object")
        if set(value) != required:
            raise OfflineArenaContractError("public task fields are not canonical")
        if _contains_private_task_key(value):
            raise OfflineArenaContractError("public task contains a private reference field")
        return cls(
            benchmark_id=value["benchmark_id"],
            sample_id=value["sample_id"],
            question=value["question"],
            choices=tuple(value["choices"]),
            answer_format=value["answer_format"],
            public_metadata=value["public_metadata"],
            asset_ids=tuple(value["asset_ids"]),
            observation_policy=value["observation_policy"],
            category=value["category"],
            submission_schema_id=value["submission_schema_id"],
        )


class ActionType(str, Enum):
    GET_TASK_CONTEXT = "GET_TASK_CONTEXT"
    LIST_ASSETS = "LIST_ASSETS"
    QUERY_STATE = "QUERY_STATE"
    OPEN_ASSET = "OPEN_ASSET"
    GET_VIEW = "GET_VIEW"
    GET_FRAME = "GET_FRAME"
    GET_FRAME_WINDOW = "GET_FRAME_WINDOW"
    CROP_REGION = "CROP_REGION"
    ZOOM_REGION = "ZOOM_REGION"
    COMPOSE_ASSETS = "COMPOSE_ASSETS"
    RECORD_EVIDENCE = "RECORD_EVIDENCE"
    LIST_EVIDENCE = "LIST_EVIDENCE"
    SUBMIT = "SUBMIT"

    @property
    def family(self) -> str:
        if self in {ActionType.GET_TASK_CONTEXT, ActionType.LIST_ASSETS}:
            return "CTX"
        if self is ActionType.QUERY_STATE:
            return "STATE"
        if self in {
            ActionType.OPEN_ASSET,
            ActionType.GET_VIEW,
            ActionType.GET_FRAME,
            ActionType.GET_FRAME_WINDOW,
            ActionType.CROP_REGION,
            ActionType.ZOOM_REGION,
            ActionType.COMPOSE_ASSETS,
        }:
            return "PER"
        if self in {ActionType.RECORD_EVIDENCE, ActionType.LIST_EVIDENCE}:
            return "EVD"
        return "SUBMIT"


@dataclass(frozen=True)
class ActionPolicy:
    """Bound the narrow JSON action surface available on a CPU worker."""

    allowed_actions: tuple[ActionType, ...] = tuple(ActionType)
    max_actions: int = 8
    media_operation_ceiling: int = 6
    max_evidence_records: int = 24
    max_evidence_refs_per_submission: int = 24
    max_action_bytes: int = 65536
    duplicate_action_policy: str = "content_addressed_cache"
    submit_precondition: str | None = None
    current_profile: str = "w2_light"
    state_revision: int = 0
    network: bool = False
    shell: bool = False
    filesystem_exploration: bool = False
    arbitrary_python: bool = False
    dynamic_imports: bool = False
    gpu: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.allowed_actions, (str, bytes, bytearray)) or not isinstance(
            self.allowed_actions, Sequence
        ):
            raise OfflineArenaContractError("allowed_actions must be a sequence")
        try:
            actions = tuple(
                item if isinstance(item, ActionType) else ActionType(item)
                for item in self.allowed_actions
            )
        except (TypeError, ValueError) as exc:
            raise OfflineArenaContractError("allowed_actions contains an unknown action") from exc
        if not actions or len(set(actions)) != len(actions):
            raise OfflineArenaContractError("allowed_actions must be non-empty and unique")
        if ActionType.SUBMIT not in actions:
            raise OfflineArenaContractError("allowed_actions must include submit")
        object.__setattr__(self, "allowed_actions", actions)
        for name, lower, upper in (
            ("max_actions", 1, 1024),
            ("media_operation_ceiling", 0, 1024),
            ("max_evidence_records", 0, 1024),
            ("max_evidence_refs_per_submission", 0, 1024),
            ("max_action_bytes", 128, 1_048_576),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
                raise OfflineArenaContractError(
                    f"{name} must be an integer from {lower} to {upper}"
                )
        if self.duplicate_action_policy != "content_addressed_cache":
            raise OfflineArenaContractError(
                "duplicate_action_policy must be content_addressed_cache"
            )
        if self.submit_precondition not in {
            None,
            ONE_SUCCESSFUL_ANSWER_BEARING_MEDIA_OBSERVATION,
            MODALITY_SUFFICIENT_EVIDENCE,
        }:
            raise OfflineArenaContractError("submit_precondition is unsupported")
        if self.current_profile not in {"direct", "w2_light"}:
            raise OfflineArenaContractError("current_profile is unsupported")
        if (
            isinstance(self.state_revision, bool)
            or not isinstance(self.state_revision, int)
            or self.state_revision < 0
        ):
            raise OfflineArenaContractError("state_revision must be non-negative")
        for name in (
            "network",
            "shell",
            "filesystem_exploration",
            "arbitrary_python",
            "dynamic_imports",
            "gpu",
        ):
            if getattr(self, name) is not False:
                raise OfflineArenaContractError(f"offline policy requires {name}=False")

    def validate(self, action: "ActionEnvelope") -> None:
        if action.action not in self.allowed_actions:
            raise OfflineArenaContractError("action is disabled by policy")
        size = len(
            json.dumps(
                action.to_dict(),
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
        if size > self.max_action_bytes:
            raise OfflineArenaContractError("action exceeds max_action_bytes")
        refs = action.evidence_refs
        if len(refs) > self.max_evidence_refs_per_submission:
            raise OfflineArenaContractError("action has too many evidence_refs")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "allowed_actions": [item.value for item in self.allowed_actions],
            "max_actions": self.max_actions,
            "media_operation_ceiling": self.media_operation_ceiling,
            "max_evidence_records": self.max_evidence_records,
            "max_evidence_refs_per_submission": (
                self.max_evidence_refs_per_submission
            ),
            "max_action_bytes": self.max_action_bytes,
            "duplicate_action_policy": self.duplicate_action_policy,
            "current_profile": self.current_profile,
            "state_revision": self.state_revision,
            "action_schemas": {
                item.value: _action_schema(item) for item in self.allowed_actions
            },
            "network": self.network,
            "shell": self.shell,
            "filesystem_exploration": self.filesystem_exploration,
            "arbitrary_python": self.arbitrary_python,
            "dynamic_imports": self.dynamic_imports,
            "gpu": self.gpu,
        }
        if self.submit_precondition is not None:
            payload["submit_precondition"] = self.submit_precondition
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ActionPolicy":
        expected = {
            "allowed_actions",
            "max_actions",
            "media_operation_ceiling",
            "max_evidence_records",
            "max_evidence_refs_per_submission",
            "max_action_bytes",
            "duplicate_action_policy",
            "current_profile",
            "state_revision",
            "action_schemas",
            "network",
            "shell",
            "filesystem_exploration",
            "arbitrary_python",
            "dynamic_imports",
            "gpu",
        }
        supplied_fields = (
            frozenset(value) if isinstance(value, Mapping) else frozenset()
        )
        accepted_fields = {
            frozenset(expected),
            frozenset((*expected, "submit_precondition")),
        }
        if supplied_fields not in accepted_fields:
            raise OfflineArenaContractError("action policy fields are not canonical")
        supplied = value["action_schemas"]
        if not isinstance(supplied, Mapping):
            raise OfflineArenaContractError("action_schemas must be an object")
        kwargs = dict(value)
        kwargs.pop("action_schemas")
        policy = cls(**kwargs)
        if supplied != policy.to_dict()["action_schemas"]:
            raise OfflineArenaContractError("action_schemas disagree with policy")
        return policy

    @property
    def submission_precondition(self) -> str | None:
        """Readable alias for the serialized submit_precondition contract."""

        return self.submit_precondition


def _action_schema(action: ActionType) -> dict[str, Any]:
    empty = {"type": "object", "additionalProperties": False, "properties": {}}
    if action in {
        ActionType.GET_TASK_CONTEXT,
        ActionType.LIST_ASSETS,
        ActionType.QUERY_STATE,
        ActionType.LIST_EVIDENCE,
    }:
        return empty
    if action in {ActionType.OPEN_ASSET, ActionType.GET_VIEW}:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["asset_id"],
            "properties": {"asset_id": {"type": "string"}},
        }
    if action is ActionType.GET_FRAME:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["asset_id"],
            "properties": {
                "asset_id": {"type": "string"},
                "frame_index": {"type": "integer", "minimum": 0},
                "timestamp_ms": {"type": "integer", "minimum": 0},
            },
        }
    if action is ActionType.GET_FRAME_WINDOW:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["asset_id"],
            "properties": {
                "asset_id": {"type": "string"},
                "start_frame": {"type": "integer", "minimum": 0},
                "end_frame": {"type": ["integer", "null"], "minimum": 0},
                "step": {"type": "integer", "minimum": 1},
                "count": {"type": "integer", "minimum": 1},
                "frame_indices": {"type": "array", "items": {"type": "integer"}},
            },
        }
    if action in {ActionType.CROP_REGION, ActionType.ZOOM_REGION}:
        properties: dict[str, Any] = {
            "asset_id": {"type": "string"},
            "region": {"type": ["array", "object", "null"]},
            "coordinate_space": {"type": "string"},
            "bbox_format": {"type": "string"},
            "max_size": {"type": ["integer", "array"]},
        }
        if action is ActionType.ZOOM_REGION:
            properties["zoom_factor"] = {"type": "number", "minimum": 1}
            properties["center"] = {"type": ["array", "null"]}
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["asset_id"],
            "properties": properties,
        }
    if action is ActionType.COMPOSE_ASSETS:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["asset_ids"],
            "properties": {
                "asset_ids": {"type": "array", "items": {"type": "string"}},
                "layout": {"type": "string"},
                "columns": {"type": ["integer", "null"]},
                "gap": {"type": "integer", "minimum": 0},
                "max_size": {"type": ["integer", "array"]},
            },
        }
    if action is ActionType.RECORD_EVIDENCE:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["modality", "provenance"],
            "properties": {
                "asset_id": {"type": ["string", "null"]},
                "modality": {"type": "string"},
                "coordinate_frame": {"type": "string"},
                "scale_type": {"type": "string"},
                "region": {"type": ["object", "array", "null"]},
                "frame": {"type": ["object", "integer", "null"]},
                "view": {"type": ["object", "string", "null"]},
                "confidence": {"type": ["number", "null"]},
                "provenance": {"type": "object"},
            },
        }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["answer", "evidence_refs", "confidence"],
        "properties": {
            "answer": {},
            "evidence_refs": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": ["number", "null"]},
        },
    }


def action_argument_schema(action: ActionType | str) -> dict[str, Any]:
    """Return a detached typed argument schema for one registered action."""

    try:
        action_type = action if isinstance(action, ActionType) else ActionType(action)
    except (TypeError, ValueError) as exc:
        raise OfflineArenaContractError("action is not supported") from exc
    return json_copy(_action_schema(action_type), "action schema")


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OfflineArenaContractError("JSON action contains a duplicate key")
        result[key] = value
    return result


@dataclass(frozen=True)
class ActionEnvelope:
    """Exactly one allowlisted operation and its typed JSON arguments."""

    action: ActionType
    arguments: dict[str, Any]

    def __post_init__(self) -> None:
        try:
            action = (
                self.action
                if isinstance(self.action, ActionType)
                else ActionType(self.action)
            )
        except (TypeError, ValueError) as exc:
            raise OfflineArenaContractError("action is not supported") from exc
        object.__setattr__(self, "action", action)
        if not isinstance(self.arguments, Mapping):
            raise OfflineArenaContractError("action.arguments must be an object")
        arguments = json_copy(self.arguments, "action.arguments")
        object.__setattr__(self, "arguments", arguments)
        self._validate_arguments()

    @property
    def action_id(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return "action-" + hashlib.sha256(payload).hexdigest()[:32]

    @property
    def evidence_refs(self) -> tuple[str, ...]:
        if self.action is not ActionType.SUBMIT:
            return ()
        return tuple(self.arguments.get("evidence_refs", ()))

    def _validate_arguments(self) -> None:
        keys = set(self.arguments)
        if self.action in {
            ActionType.GET_TASK_CONTEXT,
            ActionType.LIST_ASSETS,
            ActionType.QUERY_STATE,
            ActionType.LIST_EVIDENCE,
        }:
            if keys:
                raise OfflineArenaContractError(f"{self.action.value} takes no arguments")
            return
        if self.action in {ActionType.OPEN_ASSET, ActionType.GET_VIEW}:
            if keys != {"asset_id"}:
                raise OfflineArenaContractError(
                    f"{self.action.value} arguments must contain only asset_id"
                )
            _identifier(self.arguments["asset_id"], "asset_id")
            return
        if self.action is ActionType.GET_FRAME:
            if not {"asset_id"} <= keys or keys - {
                "asset_id", "frame_index", "timestamp_ms"
            }:
                raise OfflineArenaContractError("GET_FRAME arguments are invalid")
            if ("frame_index" in keys) == ("timestamp_ms" in keys):
                raise OfflineArenaContractError(
                    "GET_FRAME requires exactly one frame locator"
                )
            _identifier(self.arguments["asset_id"], "asset_id")
            return
        if self.action is ActionType.GET_FRAME_WINDOW:
            allowed = {
                "asset_id", "start_frame", "end_frame", "step", "count", "frame_indices"
            }
            if "asset_id" not in keys or keys - allowed:
                raise OfflineArenaContractError("GET_FRAME_WINDOW arguments are invalid")
            _identifier(self.arguments["asset_id"], "asset_id")
            return
        if self.action in {ActionType.CROP_REGION, ActionType.ZOOM_REGION}:
            allowed = {
                "asset_id", "region", "coordinate_space", "bbox_format", "max_size"
            }
            if self.action is ActionType.ZOOM_REGION:
                allowed |= {"zoom_factor", "center"}
            if "asset_id" not in keys or keys - allowed:
                raise OfflineArenaContractError(f"{self.action.value} arguments are invalid")
            if self.action is ActionType.CROP_REGION and "region" not in keys:
                raise OfflineArenaContractError("CROP_REGION requires region")
            _identifier(self.arguments["asset_id"], "asset_id")
            return
        if self.action is ActionType.COMPOSE_ASSETS:
            allowed = {"asset_ids", "layout", "columns", "gap", "max_size"}
            if "asset_ids" not in keys or keys - allowed:
                raise OfflineArenaContractError("COMPOSE_ASSETS arguments are invalid")
            _string_tuple(self.arguments["asset_ids"], "asset_ids", allow_empty=False)
            return
        if self.action is ActionType.RECORD_EVIDENCE:
            allowed = {
                "asset_id", "modality", "coordinate_frame", "scale_type", "region",
                "frame", "view", "confidence", "provenance"
            }
            if not {"modality", "provenance"} <= keys or keys - allowed:
                raise OfflineArenaContractError(
                    "RECORD_EVIDENCE arguments are invalid"
                )
            _identifier(self.arguments["modality"], "evidence.modality")
            if not isinstance(self.arguments["provenance"], Mapping):
                raise OfflineArenaContractError("evidence provenance must be an object")
            if "asset_id" in keys:
                if self.arguments["asset_id"] is not None:
                    _identifier(self.arguments["asset_id"], "asset_id")
            return
        if keys != {"answer", "evidence_refs", "confidence"}:
            raise OfflineArenaContractError(
                "SUBMIT arguments must contain answer, evidence_refs, and confidence"
            )
        _submission_answer(self.arguments["answer"])
        _string_tuple(self.arguments["evidence_refs"], "evidence_refs")
        confidence = self.arguments["confidence"]
        if confidence is not None and (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise OfflineArenaContractError("submission confidence must be in [0,1]")

    def to_dict(self) -> dict[str, Any]:
        value = {
            "action": self.action.value,
            "arguments": json_copy(self.arguments, "action.arguments"),
        }
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ActionEnvelope":
        if not isinstance(value, Mapping):
            raise OfflineArenaContractError("action envelope must be an object")
        if any(not isinstance(key, str) for key in value):
            raise OfflineArenaContractError("action envelope keys must be strings")
        required = {"action", "arguments"}
        if set(value) != required:
            raise OfflineArenaContractError("action envelope fields are not canonical")
        return cls(
            action=value["action"],
            arguments=value["arguments"],
        )

    @classmethod
    def from_json(cls, value: str) -> "ActionEnvelope":
        if not isinstance(value, str):
            raise OfflineArenaContractError("JSON action must be text")
        try:
            parsed = json.loads(value, object_pairs_hook=_strict_object_pairs)
        except OfflineArenaContractError:
            raise
        except json.JSONDecodeError:
            raise OfflineArenaContractError("action is not valid JSON") from None
        if not isinstance(parsed, Mapping):
            raise OfflineArenaContractError("JSON action must contain one object")
        return cls.from_mapping(parsed)


@dataclass(frozen=True)
class EvidenceRecord:
    """One typed evidence item with explicit provenance and visibility class."""

    evidence_id: str
    source_kind: str
    modality: str
    asset_id: str | None
    coordinate_frame: str
    scale_type: str
    region: Any
    frame: Any
    view: Any
    producer_action_id: str
    media_hash: str | None
    confidence: float | None
    provenance: dict[str, Any]

    def __post_init__(self) -> None:
        for field_name in (
            "evidence_id",
            "source_kind",
            "modality",
            "coordinate_frame",
            "scale_type",
            "producer_action_id",
        ):
            object.__setattr__(
                self, field_name, _identifier(getattr(self, field_name), field_name)
            )
        if self.source_kind not in {
            "public_raw",
            "cpu_derived",
            "model_estimated",
            "private_truth",
        }:
            raise OfflineArenaContractError("evidence.source_kind is unsupported")
        if self.coordinate_frame not in {
            "image", "camera", "world", "object", "ego", "unknown"
        }:
            raise OfflineArenaContractError("evidence.coordinate_frame is unsupported")
        if self.scale_type not in {"metric", "relative", "unknown"}:
            raise OfflineArenaContractError("evidence.scale_type is unsupported")
        if self.asset_id is not None:
            object.__setattr__(self, "asset_id", _identifier(self.asset_id, "asset_id"))
        for name in ("region", "frame", "view"):
            object.__setattr__(
                self,
                name,
                json_copy(getattr(self, name), f"evidence.{name}", public_task=True),
            )
        if self.media_hash is not None and (
            not isinstance(self.media_hash, str) or not _SHA256.fullmatch(self.media_hash)
        ):
            raise OfflineArenaContractError("evidence.media_hash must be SHA-256 or null")
        if self.confidence is not None and (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(float(self.confidence))
            or not 0.0 <= float(self.confidence) <= 1.0
        ):
            raise OfflineArenaContractError("evidence.confidence must be in [0,1]")
        if self.confidence is not None:
            object.__setattr__(self, "confidence", float(self.confidence))
        if not isinstance(self.provenance, Mapping):
            raise OfflineArenaContractError("evidence.provenance must be an object")
        object.__setattr__(
            self,
            "provenance",
            json_copy(self.provenance, "evidence.provenance", public_task=True),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "family": "EVD",
            "source_kind": self.source_kind,
            "modality": self.modality,
            "asset_id": self.asset_id,
            "coordinate_frame": self.coordinate_frame,
            "scale_type": self.scale_type,
            "region": json_copy(self.region, "evidence.region"),
            "frame": json_copy(self.frame, "evidence.frame"),
            "view": json_copy(self.view, "evidence.view"),
            "producer_action_id": self.producer_action_id,
            "media_hash": self.media_hash,
            "confidence": self.confidence,
            "provenance": json_copy(self.provenance, "evidence.provenance"),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceRecord":
        expected = {
            "evidence_id",
            "family",
            "source_kind",
            "modality",
            "asset_id",
            "coordinate_frame",
            "scale_type",
            "region",
            "frame",
            "view",
            "producer_action_id",
            "media_hash",
            "confidence",
            "provenance",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise OfflineArenaContractError("evidence fields are not canonical")
        if value.get("family") != "EVD":
            raise OfflineArenaContractError("evidence family must be EVD")
        return cls(
            evidence_id=value["evidence_id"],
            source_kind=value["source_kind"],
            modality=value["modality"],
            asset_id=value["asset_id"],
            coordinate_frame=value["coordinate_frame"],
            scale_type=value["scale_type"],
            region=value["region"],
            frame=value["frame"],
            view=value["view"],
            producer_action_id=value["producer_action_id"],
            media_hash=value["media_hash"],
            confidence=value["confidence"],
            provenance=value["provenance"],
        )


def _submission_answer(value: Any) -> str | int | float:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise OfflineArenaContractError("submission answer must be a JSON scalar")
    if isinstance(value, float) and not math.isfinite(value):
        raise OfflineArenaContractError("submission answer must be finite")
    if isinstance(value, str):
        return _text(value, "submission.answer", maximum=8192)
    return value


@dataclass(frozen=True)
class ParsedSubmission:
    """Normalized submission; evidence references are deliberately optional."""

    answer: str | int | float
    evidence_refs: tuple[str, ...] = ()
    confidence: float | None = None
    raw_submission_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "answer", _submission_answer(self.answer))
        object.__setattr__(
            self,
            "evidence_refs",
            _string_tuple(self.evidence_refs, "evidence_refs"),
        )
        if self.confidence is not None and (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(float(self.confidence))
            or not 0.0 <= float(self.confidence) <= 1.0
        ):
            raise OfflineArenaContractError("submission confidence must be in [0,1]")
        if self.confidence is not None:
            object.__setattr__(self, "confidence", float(self.confidence))
        if not self.raw_submission_hash:
            payload = json.dumps(
                {
                    "answer": self.answer,
                    "evidence_refs": list(self.evidence_refs),
                    "confidence": self.confidence,
                },
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            object.__setattr__(self, "raw_submission_hash", hashlib.sha256(payload).hexdigest())
        elif not _SHA256.fullmatch(self.raw_submission_hash):
            raise OfflineArenaContractError("raw_submission_hash must be SHA-256")

    @classmethod
    def from_envelope(cls, action: ActionEnvelope) -> "ParsedSubmission":
        if not isinstance(action, ActionEnvelope) or action.action is not ActionType.SUBMIT:
            raise OfflineArenaContractError("submission requires a submit action")
        return cls(
            answer=action.arguments["answer"],
            evidence_refs=action.evidence_refs,
            confidence=action.arguments["confidence"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "evidence_refs": list(self.evidence_refs),
            "confidence": self.confidence,
            "raw_submission_hash": self.raw_submission_hash,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ParsedSubmission":
        required = {
            "answer", "evidence_refs", "confidence", "raw_submission_hash"
        }
        if not isinstance(value, Mapping):
            raise OfflineArenaContractError("submission must be an object")
        if set(value) != required:
            raise OfflineArenaContractError("submission fields are not canonical")
        return cls(
            answer=value["answer"],
            evidence_refs=tuple(value["evidence_refs"]),
            confidence=value["confidence"],
            raw_submission_hash=value["raw_submission_hash"],
        )


@dataclass(frozen=True)
class EvaluationResult:
    """Reference-free result using the canonical W2 success distinctions."""

    episode_id: str
    benchmark_id: str
    sample_id: str
    metric: str
    score: float | None
    passed: bool | None
    prediction_normalized: str
    task_outcome: str
    evidence_refs: tuple[str, ...] = ()
    execution_success: bool = True
    contract_valid: bool = True
    submission_valid: bool = True
    score_kind: str = "diagnostic"
    official_evaluator_used: bool = False
    official_score: float | None = None
    denominator_eligible: bool = True
    error_type: str | None = None
    blocked_reason: str | None = None
    notes: str = ""
    evaluator_provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("episode_id", "benchmark_id", "sample_id", "metric"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        if self.score is not None and (
            isinstance(self.score, bool)
            or not isinstance(self.score, (int, float))
            or not math.isfinite(float(self.score))
        ):
            raise OfflineArenaContractError("result.score must be finite or null")
        if self.score is not None:
            object.__setattr__(self, "score", float(self.score))
        if self.passed is not None and not isinstance(self.passed, bool):
            raise OfflineArenaContractError("result.passed must be boolean or null")
        if not isinstance(self.prediction_normalized, str):
            raise OfflineArenaContractError("prediction_normalized must be text")
        outcomes = {
            "correct",
            "incorrect",
            "partial",
            "inconclusive",
            "unavailable",
            "not_evaluated",
        }
        if self.task_outcome not in outcomes:
            raise OfflineArenaContractError("unsupported task_outcome")
        object.__setattr__(
            self,
            "evidence_refs",
            _string_tuple(self.evidence_refs, "evidence_refs"),
        )
        for name in (
            "execution_success",
            "contract_valid",
            "submission_valid",
            "official_evaluator_used",
            "denominator_eligible",
        ):
            if not isinstance(getattr(self, name), bool):
                raise OfflineArenaContractError(f"result.{name} must be boolean")
        if self.score_kind not in {"diagnostic", "subset", "unscored"}:
            raise OfflineArenaContractError(
                "offline evaluator score_kind must be diagnostic, subset, or unscored"
            )
        if self.official_evaluator_used or self.official_score is not None:
            raise OfflineArenaContractError(
                "offline evaluator cannot claim an official score"
            )
        if self.task_outcome == "inconclusive" and (
            self.score is not None or self.denominator_eligible
        ):
            raise OfflineArenaContractError(
                "inconclusive results must be unscored and denominator-ineligible"
            )
        if self.error_type is not None:
            object.__setattr__(
                self, "error_type", _identifier(self.error_type, "error_type")
            )
        if self.blocked_reason is not None:
            object.__setattr__(
                self,
                "blocked_reason",
                _identifier(self.blocked_reason, "blocked_reason"),
            )
        if not isinstance(self.notes, str):
            raise OfflineArenaContractError("result.notes must be text")
        if not isinstance(self.evaluator_provenance, Mapping):
            raise OfflineArenaContractError("evaluator_provenance must be an object")
        object.__setattr__(
            self,
            "evaluator_provenance",
            json_copy(self.evaluator_provenance, "evaluator_provenance"),
        )

    @property
    def task_score(self) -> float | None:
        return self.score

    @property
    def task_success(self) -> bool | None:
        return self.passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": "evaluation",
            "episode_id": self.episode_id,
            "benchmark_id": self.benchmark_id,
            "sample_id": self.sample_id,
            "execution_success": self.execution_success,
            "contract_valid": self.contract_valid,
            "submission_valid": self.submission_valid,
            "task_success": self.passed,
            "task_outcome": self.task_outcome,
            "task_score": self.score,
            "metric": self.metric,
            "score": self.score,
            "passed": self.passed,
            "prediction_normalized": self.prediction_normalized,
            "score_kind": self.score_kind,
            "official_evaluator_used": self.official_evaluator_used,
            "official_score": self.official_score,
            "denominator_eligible": self.denominator_eligible,
            "evidence_refs": list(self.evidence_refs),
            "error_type": self.error_type,
            "blocked_reason": self.blocked_reason,
            "notes": self.notes,
            "evaluator_provenance": json_copy(
                self.evaluator_provenance, "evaluator_provenance"
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvaluationResult":
        expected = {
            "record_type",
            "episode_id",
            "benchmark_id",
            "sample_id",
            "execution_success",
            "contract_valid",
            "submission_valid",
            "task_success",
            "task_outcome",
            "task_score",
            "metric",
            "score",
            "passed",
            "prediction_normalized",
            "score_kind",
            "official_evaluator_used",
            "official_score",
            "denominator_eligible",
            "evidence_refs",
            "error_type",
            "blocked_reason",
            "notes",
            "evaluator_provenance",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise OfflineArenaContractError("evaluation fields are not canonical")
        if value["record_type"] != "evaluation":
            raise OfflineArenaContractError("evaluation record_type is invalid")
        if value["task_success"] != value["passed"]:
            raise OfflineArenaContractError("task_success and passed disagree")
        if value["task_score"] != value["score"]:
            raise OfflineArenaContractError("task_score and score disagree")
        return cls(
            episode_id=value["episode_id"],
            benchmark_id=value["benchmark_id"],
            sample_id=value["sample_id"],
            execution_success=value["execution_success"],
            contract_valid=value["contract_valid"],
            submission_valid=value["submission_valid"],
            passed=value["passed"],
            task_outcome=value["task_outcome"],
            score=value["score"],
            metric=value["metric"],
            prediction_normalized=value["prediction_normalized"],
            score_kind=value["score_kind"],
            official_evaluator_used=value["official_evaluator_used"],
            official_score=value["official_score"],
            denominator_eligible=value["denominator_eligible"],
            evidence_refs=tuple(value["evidence_refs"]),
            error_type=value["error_type"],
            blocked_reason=value["blocked_reason"],
            notes=value["notes"],
            evaluator_provenance=value["evaluator_provenance"],
        )


@dataclass(frozen=True)
class Observation:
    """One public environment view with no evaluator or reference handle."""

    episode_id: str
    state_revision: int
    lifecycle_state: EpisodeLifecycleState
    task_summary: PublicTask
    visible_assets: tuple[AssetDescriptor, ...]
    new_evidence: tuple[EvidenceRecord, ...]
    current_evidence_index: tuple[str, ...]
    action_result: dict[str, Any]
    available_actions: tuple[ActionType, ...]
    remaining_budget: dict[str, int]
    terminal: bool
    safe_error: dict[str, Any] | None = None
    valid_submission_evidence_refs: tuple[str, ...] = ()
    model_visible_media: tuple[dict[str, Any], ...] = ()
    evidence_binding_contract_hash: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "episode_id", _identifier(self.episode_id, "episode_id"))
        try:
            state = (
                self.lifecycle_state
                if isinstance(self.lifecycle_state, EpisodeLifecycleState)
                else EpisodeLifecycleState(self.lifecycle_state)
            )
        except (TypeError, ValueError) as exc:
            raise OfflineArenaContractError("observation state is unknown") from exc
        object.__setattr__(self, "lifecycle_state", state)
        if (
            isinstance(self.state_revision, bool)
            or not isinstance(self.state_revision, int)
            or self.state_revision < 0
        ):
            raise OfflineArenaContractError("state_revision must be non-negative")
        if not isinstance(self.task_summary, PublicTask):
            raise OfflineArenaContractError("task_summary must be PublicTask")
        assets = tuple(self.visible_assets)
        evidence = tuple(self.new_evidence)
        if not all(isinstance(item, AssetDescriptor) for item in assets):
            raise OfflineArenaContractError("visible_assets must be typed")
        if not all(isinstance(item, EvidenceRecord) for item in evidence):
            raise OfflineArenaContractError("new_evidence must be typed")
        if any(item.source_kind == "private_truth" for item in evidence):
            raise OfflineArenaContractError("private_truth evidence cannot enter Observation")
        object.__setattr__(self, "visible_assets", assets)
        object.__setattr__(self, "new_evidence", evidence)
        object.__setattr__(
            self,
            "current_evidence_index",
            _string_tuple(self.current_evidence_index, "current_evidence_index"),
        )
        if not isinstance(self.action_result, Mapping):
            raise OfflineArenaContractError("action_result must be an object")
        object.__setattr__(
            self,
            "action_result",
            json_copy(self.action_result, "action_result", public_task=True),
        )
        try:
            actions = tuple(
                item if isinstance(item, ActionType) else ActionType(item)
                for item in self.available_actions
            )
        except (TypeError, ValueError) as exc:
            raise OfflineArenaContractError("observation has an unknown action") from exc
        if len(set(actions)) != len(actions):
            raise OfflineArenaContractError("available_actions must be unique")
        object.__setattr__(self, "available_actions", actions)
        if not isinstance(self.remaining_budget, Mapping):
            raise OfflineArenaContractError("remaining_budget must be an object")
        budget = dict(self.remaining_budget)
        if any(
            not isinstance(key, str)
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for key, value in budget.items()
        ):
            raise OfflineArenaContractError(
                "remaining_budget must contain non-negative counters"
            )
        object.__setattr__(self, "remaining_budget", budget)
        if not isinstance(self.terminal, bool):
            raise OfflineArenaContractError("terminal must be boolean")
        if self.terminal != state.terminal:
            raise OfflineArenaContractError("terminal flag disagrees with lifecycle")
        if self.safe_error is not None and not isinstance(self.safe_error, Mapping):
            raise OfflineArenaContractError("safe_error must be an object or null")
        object.__setattr__(
            self,
            "safe_error",
            None
            if self.safe_error is None
            else json_copy(self.safe_error, "safe_error", public_task=True),
        )
        valid_refs = _string_tuple(
            self.valid_submission_evidence_refs,
            "valid_submission_evidence_refs",
        )
        if len(set(valid_refs)) != len(valid_refs):
            raise OfflineArenaContractError(
                "valid_submission_evidence_refs must be unique"
            )
        media = json_copy(
            self.model_visible_media,
            "model_visible_media",
            public_task=True,
        )
        if not isinstance(media, list):
            raise OfflineArenaContractError("model_visible_media must be a sequence")
        media_fields = {
            "asset_id",
            "evidence_ref",
            "modality",
            "view_id",
            "frame_id",
            "source_kind",
        }
        for item in media:
            if not isinstance(item, Mapping) or set(item) != media_fields:
                raise OfflineArenaContractError(
                    "model_visible_media fields are not canonical"
                )
            if item["evidence_ref"] not in valid_refs:
                raise OfflineArenaContractError(
                    "model-visible evidence ref is not a valid submission ref"
                )
            if item["source_kind"] not in {"public_raw", "cpu_derived"}:
                raise OfflineArenaContractError(
                    "model-visible media must be public evidence"
                )
            for key in ("asset_id", "evidence_ref", "modality"):
                _identifier(item[key], f"model_visible_media.{key}")
            for key in ("view_id", "frame_id"):
                if item[key] is not None:
                    _identifier(item[key], f"model_visible_media.{key}")
        binding_hash = self.evidence_binding_contract_hash
        if binding_hash is not None and (
            not isinstance(binding_hash, str) or not _SHA256.fullmatch(binding_hash)
        ):
            raise OfflineArenaContractError(
                "evidence_binding_contract_hash must be SHA-256 or null"
            )
        object.__setattr__(self, "valid_submission_evidence_refs", valid_refs)
        object.__setattr__(self, "model_visible_media", tuple(media))
        object.__setattr__(self, "evidence_binding_contract_hash", binding_hash)

    @property
    def evidence_refs(self) -> tuple[str, ...]:
        return self.current_evidence_index

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": "observation",
            "episode_id": self.episode_id,
            "state_revision": self.state_revision,
            "lifecycle_state": self.lifecycle_state.value,
            "task_summary": self.task_summary.to_dict(),
            "visible_assets": [item.to_dict() for item in self.visible_assets],
            "new_evidence": [item.to_dict() for item in self.new_evidence],
            "current_evidence_index": list(self.current_evidence_index),
            "valid_submission_evidence_refs": list(
                self.valid_submission_evidence_refs
            ),
            "model_visible_media": json_copy(
                self.model_visible_media, "model_visible_media"
            ),
            "evidence_binding_contract_hash": self.evidence_binding_contract_hash,
            "action_result": json_copy(self.action_result, "action_result"),
            "available_actions": [item.value for item in self.available_actions],
            "remaining_budget": dict(self.remaining_budget),
            "terminal": self.terminal,
            "safe_error": (
                None
                if self.safe_error is None
                else json_copy(self.safe_error, "safe_error")
            ),
        }


__all__ = [
    "MEDIA_OBSERVATION_REQUIRED_BEFORE_SUBMIT",
    "MODALITY_SUFFICIENT_EVIDENCE",
    "ONE_SUCCESSFUL_ANSWER_BEARING_MEDIA_OBSERVATION",
    "STATE_SCHEMA_VERSION",
    "ActionEnvelope",
    "ActionPolicy",
    "ActionType",
    "AssetDescriptor",
    "EvaluationResult",
    "EvidenceRecord",
    "Observation",
    "OfflineArenaContractError",
    "ParsedSubmission",
    "PublicTask",
    "action_argument_schema",
    "json_copy",
]
