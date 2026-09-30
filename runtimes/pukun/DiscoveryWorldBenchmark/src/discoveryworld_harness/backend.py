from __future__ import annotations

import contextlib
import hashlib
import io
import os
from dataclasses import dataclass
from typing import Any

from .config import HarnessConfig
from .manifest import TaskRecord
from .verifier import DiscoveryWorldNativeVerifier, VerificationResult, empty_verification


@dataclass(frozen=True)
class TaskState:
    task: TaskRecord
    observation: dict[str, Any]
    known_actions: dict[str, Any]
    teleport_locations: dict[str, Any]
    verification: VerificationResult


@dataclass(frozen=True)
class StepResult:
    native_action_json: dict[str, Any] | None
    observation_before: dict[str, Any]
    observation_after: dict[str, Any]
    action_response: dict[str, Any] | None
    tick_response: dict[str, Any] | None
    verification: dict[str, Any]
    valid_action: bool
    error: dict[str, Any] | None = None
    teleport_used: bool = False

    @property
    def success(self) -> bool:
        return bool(self.verification.get("success"))

    @property
    def completed(self) -> bool:
        return bool(self.verification.get("completed"))

    @property
    def score(self) -> float:
        return float(self.verification.get("scoreNormalized") or 0.0)


class DiscoveryWorldBackend:
    def __init__(self, config: HarnessConfig):
        self.config = config
        self.api: Any | None = None
        self.task: TaskRecord | None = None
        self.observation_raw: dict[str, Any] = {}
        self.verification = empty_verification()

    def reset_task(self, task: TaskRecord) -> TaskState:
        self.close()
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        from discoveryworld.DiscoveryWorldAPI import DiscoveryWorldAPI

        self.task = task
        self.api = DiscoveryWorldAPI(threadID=_thread_id(task.task_id))
        loaded = self._official_call(
            self.api.loadScenario,
            task.scenario_name,
            task.difficulty,
            randomSeed=task.seed,
            numUserAgents=1,
        )
        if not loaded:
            raise RuntimeError(f"Failed to load DiscoveryWorld scenario: {task}")
        self.observation_raw = self._official_call(self.api.getAgentObservation, 0)
        goal_text = _extract_goal_text(self.observation_raw)
        if goal_text:
            task.goal_text = goal_text
        self.verification = self.check_success()
        return TaskState(
            task=task,
            observation=self.observe(),
            known_actions=self.list_known_actions(),
            teleport_locations=self.list_teleport_locations(),
            verification=self.verification,
        )

    def observe(self) -> dict[str, Any]:
        if os.environ.get('ARENA_CASE_STUDY_DIR'):
            from embodied_harness.case_study_recording import publish_observation
            publish_observation(self.observation_raw.get('vision', {}), 'discoveryworld.vision')
        return summarize_observation(self.observation_raw)

    def list_known_actions(self, limited: bool = False) -> dict[str, Any]:
        self._require_api()
        return dict(self._official_call(self.api.listKnownActions, limited=limited))

    def list_teleport_locations(self) -> dict[str, Any]:
        self._require_api()
        locations = self._official_call(self.api.listTeleportLocationsDict)
        return dict(locations or {})

    def perform_action(self, action_json: dict[str, Any]) -> StepResult:
        self._require_api()
        observation_before = self.observe()
        action_response = self._official_call(self.api.performAgentAction, 0, dict(action_json))
        tick_response = self._official_call(self.api.tick)
        self.observation_raw = self._official_call(self.api.getAgentObservation, 0)
        self.verification = self.check_success()
        valid = bool(action_response.get("success")) and not action_response.get("errors")
        return StepResult(
            native_action_json=dict(action_json),
            observation_before=observation_before,
            observation_after=self.observe(),
            action_response=action_response,
            tick_response=tick_response,
            verification=self.verification.to_agent_dict(),
            valid_action=valid,
            error=None if valid else {"kind": "invalid_action", "detail": action_response},
            teleport_used=action_json.get("action") == "TELEPORT_TO_LOCATION",
        )

    def tick(self) -> StepResult:
        self._require_api()
        observation_before = self.observe()
        tick_response = self._official_call(self.api.tick)
        self.observation_raw = self._official_call(self.api.getAgentObservation, 0)
        self.verification = self.check_success()
        return StepResult(
            native_action_json=None,
            observation_before=observation_before,
            observation_after=self.observe(),
            action_response=None,
            tick_response=tick_response,
            verification=self.verification.to_agent_dict(),
            valid_action=bool(tick_response.get("success")),
            error=None if tick_response.get("success") else {"kind": "tick_failed", "detail": tick_response},
        )

    def invalid_result(self, error: dict[str, Any]) -> StepResult:
        return StepResult(
            native_action_json=None,
            observation_before=self.observe(),
            observation_after=self.observe(),
            action_response=None,
            tick_response=None,
            verification=self.verification.to_agent_dict(),
            valid_action=False,
            error=error,
        )

    def check_success(self) -> VerificationResult:
        if self.api is None:
            return empty_verification()
        self.verification = DiscoveryWorldNativeVerifier.from_api(self.api)
        return self.verification

    def close(self) -> None:
        self.api = None
        self.observation_raw = {}
        self.verification = empty_verification()

    def _require_api(self) -> None:
        if self.api is None:
            raise RuntimeError("DiscoveryWorldBackend.reset_task must be called first.")

    def _official_call(self, func, *args: Any, **kwargs: Any) -> Any:
        if not self.config.suppress_official_output:
            return func(*args, **kwargs)
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            return func(*args, **kwargs)


def summarize_observation(observation: dict[str, Any]) -> dict[str, Any]:
    ui = observation.get("ui", {}) if isinstance(observation, dict) else {}
    return {
        "errors": observation.get("errors", []) if isinstance(observation, dict) else [],
        "world_steps": ui.get("world_steps"),
        "taskProgress": _jsonable(ui.get("taskProgress", [])),
        "agentLocation": _jsonable(ui.get("agentLocation", {})),
        "accessibleEnvironmentObjects": [_object_summary(obj) for obj in ui.get("accessibleEnvironmentObjects", [])],
        "inventoryObjects": [_object_summary(obj) for obj in ui.get("inventoryObjects", [])],
        "nearbyObjects": _summarize_nearby(ui.get("nearbyObjects", {})),
        "nearbyAgents": _jsonable(ui.get("nearbyAgents", {})),
        "dialog_box": _jsonable(ui.get("dialog_box", {"is_in_dialog": False})),
        "discoveryFeed": _jsonable(ui.get("discoveryFeed", {})),
        "lastActionMessage": ui.get("lastActionMessage", ""),
        "extended_action_message": ui.get("extended_action_message", ""),
    }


def current_objects(observation: dict[str, Any]) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for scope in ("accessibleEnvironmentObjects", "inventoryObjects"):
        for obj in observation.get(scope, []):
            item = dict(obj)
            item["scope"] = scope
            objects.append(item)
    return objects


def _object_summary(obj: dict[str, Any]) -> dict[str, Any]:
    return {
        "uuid": obj.get("uuid"),
        "name": obj.get("name"),
        "description": obj.get("description"),
    }


def _summarize_nearby(nearby: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(nearby, dict):
        return {}
    objects = nearby.get("objects", {})
    summarized: dict[str, Any] = {
        "distance": nearby.get("distance"),
        "note": nearby.get("note"),
        "objects": {},
    }
    if isinstance(objects, dict):
        for direction, items in objects.items():
            summarized["objects"][direction] = [_object_summary(item) | {"distance": item.get("distance")} for item in items]
    return summarized


def _extract_goal_text(observation: dict[str, Any]) -> str:
    ui = observation.get("ui", {}) if isinstance(observation, dict) else {}
    progress = ui.get("taskProgress") or []
    descriptions = [str(item.get("description", "")).strip() for item in progress if item.get("description")]
    return "\n".join(descriptions).strip()


def _thread_id(task_id: str) -> int:
    digest = hashlib.sha1(task_id.encode("utf-8")).hexdigest()
    return 1000 + (int(digest[:8], 16) % 800000)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value
