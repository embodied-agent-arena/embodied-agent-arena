from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ROB_ROOT = ROOT.parent
EXTERNAL_ROOT = Path(os.environ.get("EMBODIED_ARENA_EXTERNAL_ROOT", ROB_ROOT.parents[1] / "external"))
ALFWORLD_SRC = Path(os.environ.get("ALFWORLD_ROOT", EXTERNAL_ROOT / "upstreams/alfworld"))
REWARD_CONFIG = ALFWORLD_SRC / "alfworld" / "agents" / "config" / "rewards.json"


@dataclass(frozen=True)
class StepResult:
    primitive: str
    native_action: dict[str, Any] | None
    observation_before: dict[str, Any]
    observation_after: dict[str, Any]
    verification: dict[str, Any]
    valid_action: bool
    error: dict[str, Any] | None = None

    @property
    def success(self) -> bool:
        return bool(self.verification.get("success"))

    @property
    def completed(self) -> bool:
        return bool(self.verification.get("completed"))

    @property
    def score(self) -> float:
        return float(self.verification.get("score") or 0.0)


class AlfredOfficialBackend:
    def __init__(
        self,
        *,
        screen_size: int = 300,
        quality: str = "MediumCloseFitShadows",
        reward_config: Path = REWARD_CONFIG,
    ):
        self.screen_size = screen_size
        self.quality = quality
        self.reward_config = reward_config
        self.env: Any | None = None
        self.task: dict[str, Any] | None = None
        self.traj: dict[str, Any] | None = None
        self.verification = empty_verification()

    def reset_task(self, task: dict[str, Any]) -> dict[str, Any]:
        self.close()
        self.task = dict(task)
        self.traj = _load_json(Path(task["traj_path"]))
        scene = self.traj["scene"]
        scene_name = scene.get("floor_plan") or f"FloorPlan{scene['scene_num']}"

        if str(ALFWORLD_SRC) not in sys.path:
            sys.path.insert(0, str(ALFWORLD_SRC))
        from alfworld.env.thor_env import ThorEnv

        self.env = ThorEnv(
            build_path=os.environ.get("ALFWORLD_THOR_EXECUTABLE") or None,
            player_screen_height=self.screen_size,
            player_screen_width=self.screen_size,
            quality=self.quality,
            save_frames_to_disk=False,
            smooth_nav=False,
        )
        self.env.reset(scene_name)
        self.env.restore_scene(
            scene.get("object_poses", []),
            scene.get("object_toggles", []),
            scene.get("dirty_and_empty", False),
        )
        self.env.step(dict(scene["init_action"]))
        self.env.set_task(
            self.traj,
            SimpleNamespace(reward_config=str(self.reward_config)),
            reward_type="dense",
        )
        self.verification = self.check_success()
        return {"task": self.task, "observation": self.observe(), "verification": self.verification}

    def observe(self) -> dict[str, Any]:
        self._require_env()
        event = self.env.last_event
        return {
            "scene": event.metadata.get("sceneName"),
            "agent": _agent_summary(event.metadata.get("agent", {})),
            "lastActionSuccess": event.metadata.get("lastActionSuccess"),
            "errorMessage": event.metadata.get("errorMessage"),
            "frame_shape": _shape(getattr(event, "frame", None)),
            "visible_objects": [_safe_object(obj) for obj in self.visible_objects()],
            "inventory": [_safe_object(obj) for obj in self.inventory_objects()],
        }

    def save_frame(self, task_id: str, label: str = "frame") -> dict[str, Any]:
        self._require_env()
        frame = getattr(self.env.last_event, "frame", None)
        if frame is None:
            return {"saved": False, "path": None, "shape": None}
        frame_dir = ROOT / "outputs" / "frames" / _safe_name(task_id)
        frame_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(f"{task_id}:{label}:{len(list(frame_dir.glob('*.png')))}".encode("utf-8")).hexdigest()[:10]
        path = frame_dir / f"{_safe_name(label)}_{digest}.png"
        _save_image(path, frame)
        return {"saved": True, "path": str(path), "shape": _shape(frame)}

    def all_objects(self) -> list[dict[str, Any]]:
        self._require_env()
        return list(self.env.last_event.metadata.get("objects", []))

    def visible_objects(self) -> list[dict[str, Any]]:
        return [obj for obj in self.all_objects() if obj.get("visible")]

    def inventory_objects(self) -> list[dict[str, Any]]:
        self._require_env()
        return list(self.env.last_event.metadata.get("inventoryObjects", []))

    def step(self, primitive: str, native_action: dict[str, Any]) -> StepResult:
        self._require_env()
        observation_before = self.observe()
        event = self.env.step(dict(native_action))
        valid = bool(event.metadata.get("lastActionSuccess"))
        self.verification = self.check_success()
        error = None
        if not valid:
            error = {"kind": "native_action_failed", "message": event.metadata.get("errorMessage", "")}
        return StepResult(
            primitive=primitive,
            native_action=dict(native_action),
            observation_before=observation_before,
            observation_after=self.observe(),
            verification=dict(self.verification),
            valid_action=valid,
            error=error,
        )

    def invalid_result(self, primitive: str, error: dict[str, Any]) -> StepResult:
        self.verification = self.check_success()
        obs = self.observe() if self.env is not None else {}
        return StepResult(
            primitive=primitive,
            native_action=None,
            observation_before=obs,
            observation_after=obs,
            verification=dict(self.verification),
            valid_action=False,
            error=error,
        )

    def check_success(self) -> dict[str, Any]:
        if self.env is None:
            return empty_verification()
        met, total = self.env.get_goal_conditions_met()
        success = bool(self.env.get_goal_satisfied())
        return {
            "success": success,
            "completed": success,
            "score": float(met) / float(total or 1),
            "goal_conditions_met": int(met),
            "goal_conditions_total": int(total),
        }

    def close(self) -> None:
        if self.env is not None:
            for method_name in ("stop", "close"):
                method = getattr(self.env, method_name, None)
                if method is not None:
                    try:
                        method()
                    except Exception:
                        pass
                    break
        self.env = None
        self.task = None
        self.traj = None
        self.verification = empty_verification()

    def _require_env(self) -> None:
        if self.env is None:
            raise RuntimeError("AlfredOfficialBackend.reset_task must be called first.")


def empty_verification() -> dict[str, Any]:
    return {
        "success": False,
        "completed": False,
        "score": 0.0,
        "goal_conditions_met": 0,
        "goal_conditions_total": 0,
    }


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_object(obj: dict[str, Any]) -> dict[str, Any]:
    return {
        "objectId": obj.get("objectId"),
        "objectType": obj.get("objectType"),
        "name": obj.get("name"),
        "visible": obj.get("visible"),
        "distance": _round(obj.get("distance")),
        "pickupable": obj.get("pickupable"),
        "openable": obj.get("openable"),
        "isOpen": obj.get("isOpen"),
        "receptacle": obj.get("receptacle"),
        "toggleable": obj.get("toggleable"),
        "isToggled": obj.get("isToggled"),
        "isPickedUp": obj.get("isPickedUp"),
        "sliceable": obj.get("sliceable"),
        "isSliced": obj.get("isSliced"),
        "dirtyable": obj.get("dirtyable"),
        "isDirty": obj.get("isDirty"),
        "parentReceptacles": obj.get("parentReceptacles"),
        "position": obj.get("position"),
        "rotation": obj.get("rotation"),
    }


def _agent_summary(agent: dict[str, Any]) -> dict[str, Any]:
    return {
        "position": agent.get("position"),
        "rotation": agent.get("rotation"),
        "cameraHorizon": agent.get("cameraHorizon"),
        "isStanding": agent.get("isStanding"),
    }


def _shape(frame: Any) -> list[int] | None:
    shape = getattr(frame, "shape", None)
    return [int(value) for value in shape] if shape is not None else None


def _round(value: Any) -> Any:
    return round(float(value), 3) if isinstance(value, (float, int)) else value


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(value))
    return cleaned[:120] or "frame"


def _save_image(path: Path, frame: Any) -> None:
    try:
        from PIL import Image

        Image.fromarray(frame).save(path)
        if __import__('os').environ.get('ARENA_CASE_STUDY_DIR'):
            from embodied_harness.case_study_recording import publish_image
            publish_image(frame, 'thor.egocentric_rgb', source_path=path)
        return
    except ModuleNotFoundError:
        pass
    try:
        import cv2

        cv2.imwrite(str(path), frame[:, :, ::-1])
        return
    except Exception as exc:
        raise RuntimeError("Saving RGB frames requires Pillow or OpenCV in the visual runtime.") from exc
