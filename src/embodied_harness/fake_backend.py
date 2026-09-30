from __future__ import annotations

from copy import deepcopy
from typing import Any

from .backend import EmbodiedBackend
from .schemas import EpisodeTrace, Observation, PrimitiveCard, PrimitiveResult, TaskSpec, VerificationResult


class FakeBackend(EmbodiedBackend):
    """Deterministic toy backend used to lock the M0 harness contract."""

    def __init__(self, tasks: dict[str, TaskSpec] | None = None) -> None:
        self._tasks = tasks or self._default_tasks()
        self._task: TaskSpec | None = None
        self._state: dict[str, Any] = {}
        self._trace: EpisodeTrace | None = None

    @staticmethod
    def _default_tasks() -> dict[str, TaskSpec]:
        tasks: dict[str, TaskSpec] = {}
        task = TaskSpec(
            task_id="fake_place_red_cube",
            source="fake_backend",
            instruction="Place the red cube on the blue pad.",
            goal={"relation": "on", "object_id": "red_cube", "target_id": "blue_pad"},
            initial_state={
                "holding": None,
                "evidence": {},
                "objects": {
                    "red_cube": {"label": "red cube", "location": "table", "visible": True},
                    "blue_pad": {"label": "blue pad", "location": "table", "visible": True},
                },
            },
            budgets={"primitive_calls": 10, "verifier_calls": 5},
            tags=["fake", "m0", "pick_place", "verifier"],
            allowed_primitive_levels=["L1", "L2", "L3"],
            metadata={"oracle_leakage_level": "L5"},
        )
        tasks[task.task_id] = task

        colors = ["red", "green", "yellow", "purple", "orange"]
        shapes = ["cube", "cylinder", "block", "peg"]
        for index, (color, shape) in enumerate((color, shape) for color in colors for shape in shapes):
            object_id = f"{color}_{shape}_{index:02d}"
            target_id = f"target_pad_{index:02d}"
            task_id = f"fake_place_{color}_{shape}_{index:02d}"
            tasks[task_id] = TaskSpec(
                task_id=task_id,
                source="fake_backend",
                instruction=f"Place the {color} {shape} on the silver pad {index}.",
                goal={"relation": "on", "object_id": object_id, "target_id": target_id},
                initial_state={
                    "holding": None,
                    "evidence": {},
                    "objects": {
                        object_id: {"label": f"{color} {shape}", "location": "table", "visible": True},
                        target_id: {"label": f"silver pad {index}", "location": "table", "visible": True},
                    },
                },
                budgets={"primitive_calls": 10, "verifier_calls": 5},
                tags=["fake", "m0", "pick_place", "generated"],
                allowed_primitive_levels=["L1", "L2", "L3"],
                metadata={"oracle_leakage_level": "L5", "generated_index": index},
            )
        return tasks

    def list_task_ids(self) -> list[str]:
        return sorted(self._tasks)

    def reset(self, task_id: str, seed: int | None = None, config: dict[str, Any] | None = None) -> TaskSpec:
        if task_id not in self._tasks:
            raise KeyError(f"Unknown fake task: {task_id}")
        self._task = self._tasks[task_id]
        self._state = deepcopy(self._task.initial_state)
        self._state["seed"] = seed
        self._state["config"] = config or {}
        self._trace = EpisodeTrace(task_id=task_id)
        self.record_event("reset", {"seed": seed, "config": config or {}, "task": self._task.to_dict()})
        return self._task

    def observe(self) -> Observation:
        self._require_task()
        visible_objects = {
            object_id: {
                "label": obj["label"],
                "location": obj["location"],
                "visible": obj["visible"],
            }
            for object_id, obj in self._state["objects"].items()
            if obj.get("visible", False)
        }
        obs = Observation(
            step=len(self.get_trace().events),
            data={
                "instruction": self._task.instruction if self._task else "",
                "holding": self._state.get("holding"),
                "visible_objects": visible_objects,
            },
        )
        self.record_event("observe", obs.to_dict())
        return obs

    def list_primitives(self, level: str | None = None) -> list[PrimitiveCard]:
        cards = [
            PrimitiveCard(
                name="find_object",
                capability_tags=["perception", "grounding"],
                input_schema={"query": "str"},
                output_schema={"object_id": "str"},
                cost={"primitive_calls": 1},
                failure_modes=["not_found", "ambiguous_query"],
                abstraction_level="L1",
                description="Ground a text query to a visible object id.",
            ),
            PrimitiveCard(
                name="pick",
                capability_tags=["execution", "manipulation"],
                input_schema={"object_id": "str"},
                output_schema={"holding": "str"},
                preconditions=["object must be visible", "gripper must be empty"],
                side_effects=["updates holding", "updates object location"],
                cost={"primitive_calls": 1},
                failure_modes=["object_not_visible", "already_holding"],
                abstraction_level="L3",
                description="Pick up a visible object.",
            ),
            PrimitiveCard(
                name="place",
                capability_tags=["execution", "manipulation", "spatial_relation"],
                input_schema={"object_id": "str", "target_id": "str", "relation": "str"},
                output_schema={"location": "str"},
                preconditions=["agent must be holding object", "target must exist"],
                side_effects=["updates holding", "updates object location"],
                cost={"primitive_calls": 1},
                failure_modes=["not_holding_object", "target_not_found"],
                abstraction_level="L3",
                description="Place a held object in a relation to a target object.",
            ),
            PrimitiveCard(
                name="query_state",
                capability_tags=["state", "debug"],
                output_schema={"state": "dict"},
                cost={"primitive_calls": 1},
                leakage_risk="L3_state",
                abstraction_level="L2",
                description="Return current symbolic state for debugging.",
            ),
            PrimitiveCard(
                name="write_evidence",
                capability_tags=["evidence", "memory"],
                input_schema={"key": "str", "value": "object"},
                output_schema={"artifact_id": "str"},
                cost={"primitive_calls": 1},
                abstraction_level="L2",
                description="Store an evidence artifact in the episode trace.",
            ),
        ]
        if level is not None:
            cards = [card for card in cards if card.abstraction_level == level]
        self.record_event("list_primitives", {"level": level, "count": len(cards)})
        return cards

    def call_primitive(self, name: str, **kwargs: Any) -> PrimitiveResult:
        self._require_task()
        handler = getattr(self, f"_primitive_{name}", None)
        if handler is None:
            result = PrimitiveResult(name=name, ok=False, error=f"Unknown primitive: {name}")
        else:
            result = handler(**kwargs)
        self.record_event("primitive_call", {"name": name, "kwargs": kwargs, "result": result.to_dict()})
        return result

    def verify(self, scope: str = "task", **kwargs: Any) -> VerificationResult:
        self._require_task()
        if scope == "task":
            goal = self._task.goal if self._task else {}
            object_id = goal.get("object_id")
            target_id = goal.get("target_id")
            relation = goal.get("relation", "on")
            ok = self._state["objects"][object_id]["location"] == f"{relation}:{target_id}"
            result = VerificationResult(
                ok=ok,
                scope=scope,
                message="goal satisfied" if ok else "goal not yet satisfied",
                metrics={"success": float(ok)},
            )
        elif scope == "evidence":
            ok = bool(self._state.get("evidence"))
            result = VerificationResult(
                ok=ok,
                scope=scope,
                message="evidence exists" if ok else "no evidence recorded",
                metrics={"evidence_count": len(self._state.get("evidence", {}))},
            )
        else:
            result = VerificationResult(ok=False, scope=scope, message=f"Unknown verification scope: {scope}")
        self.get_trace().final_status = "success" if result.ok and scope == "task" else self.get_trace().final_status
        self.record_event("verifier_call", result.to_dict())
        return result

    def get_trace(self) -> EpisodeTrace:
        if self._trace is None:
            raise RuntimeError("Backend has not been reset.")
        return self._trace

    def _primitive_find_object(self, query: str) -> PrimitiveResult:
        query_norm = query.lower().replace("_", " ").strip()
        matches = [
            object_id
            for object_id, obj in self._state["objects"].items()
            if obj.get("visible") and (query_norm in obj["label"] or query_norm == object_id.replace("_", " "))
        ]
        if len(matches) != 1:
            return PrimitiveResult(
                name="find_object",
                ok=False,
                error="not_found" if not matches else "ambiguous_query",
                metadata={"matches": matches},
            )
        return PrimitiveResult(name="find_object", ok=True, output={"object_id": matches[0]})

    def _primitive_pick(self, object_id: str) -> PrimitiveResult:
        obj = self._state["objects"].get(object_id)
        if obj is None:
            return PrimitiveResult(name="pick", ok=False, error="object_not_found")
        if not obj.get("visible"):
            return PrimitiveResult(name="pick", ok=False, error="object_not_visible")
        if self._state.get("holding") is not None:
            return PrimitiveResult(name="pick", ok=False, error="already_holding")
        self._state["holding"] = object_id
        obj["location"] = "held"
        return PrimitiveResult(name="pick", ok=True, output={"holding": object_id})

    def _primitive_place(self, object_id: str, target_id: str, relation: str = "on") -> PrimitiveResult:
        if target_id not in self._state["objects"]:
            return PrimitiveResult(name="place", ok=False, error="target_not_found")
        if self._state.get("holding") != object_id:
            return PrimitiveResult(name="place", ok=False, error="not_holding_object")
        self._state["holding"] = None
        self._state["objects"][object_id]["location"] = f"{relation}:{target_id}"
        return PrimitiveResult(name="place", ok=True, output={"object_id": object_id, "location": f"{relation}:{target_id}"})

    def _primitive_query_state(self) -> PrimitiveResult:
        return PrimitiveResult(name="query_state", ok=True, output={"state": deepcopy(self._state)})

    def _primitive_write_evidence(self, key: str, value: Any) -> PrimitiveResult:
        artifact_id = f"evidence:{key}"
        payload = {"key": key, "value": value}
        self._state.setdefault("evidence", {})[key] = value
        self.get_trace().add_artifact(artifact_id, payload)
        return PrimitiveResult(name="write_evidence", ok=True, output={"artifact_id": artifact_id}, artifacts=[artifact_id])

    def _require_task(self) -> None:
        if self._task is None:
            raise RuntimeError("Call reset() before using the backend.")
