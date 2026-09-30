from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from .backend import AlfredOfficialBackend, StepResult
from .manifest import public_task_context
from .trace import TraceWriter


FAMILIES = {
    "get_task_context": "CTX",
    "list_actions": "CTX",
    "observe": "PER",
    "get_frame": "PER",
    "inspect_current_view": "PER",
    "detect_objects": "DET",
    "remember_visible_objects": "EVD",
    "recall_visible_objects": "EVD",
    "read_search_memory": "EVD",
    "read_observed_spatial_map": "STATE",
    "scan_scene": "PER",
    "search_scene": "PER",
    "explore_room": "PER",
    "locate_object": "PER",
    "ground_object": "DET",
    "query_object_state": "STATE",
    "query_inventory": "STATE",
    "move_ahead": "NAV",
    "rotate": "NAV",
    "look": "NAV",
    "approach_object": "NAV",
    "open_object": "HACT",
    "close_object": "HACT",
    "pickup_object": "HACT",
    "put_object": "HACT",
    "toggle_object": "HACT",
    "slice_object": "HACT",
    "pickup_located_object": "HACT",
    "place_held_object": "HACT",
    "toggle_located_object": "HACT",
    "open_located_object": "HACT",
    "write_evidence": "EVD",
    "read_evidence": "EVD",
    "check_progress_public": "VERIFY",
    "check_success": "VERIFY",
    "agent_code_generated": "CTX",
    "code_execution_started": "CTX",
    "code_turn_feedback_written": "CTX",
    "code_execution_finished": "VERIFY",
}

ROTATE_ACTIONS = {"left": "RotateLeft", "right": "RotateRight"}
LOOK_ACTIONS = {"up": "LookUp", "down": "LookDown"}


class AlfredOfficialPrimitives:
    def __init__(self, backend: AlfredOfficialBackend, trace: TraceWriter, task: dict[str, Any], suite: str):
        self.backend = backend
        self.trace = trace
        self.task = task
        self.suite = suite
        self.evidence: dict[str, Any] = {}
        self.metrics = {
            "env_steps": 0,
            "invalid_action_count": 0,
            "wrapper_no_match_count": 0,
            "wrapper_ambiguity_count": 0,
        }
        self.provenance: dict[str, int | list[dict[str, Any]]] = {}
        self.visual_memory: dict[str, dict[str, Any]] = {}
        self.memory_tick = 0
        self.active_max_env_steps: int | None = None
        self.public_frame_dir: Path | None = None
        self.visited_cells: dict[str, dict[str, Any]] = {}
        self.blocked_moves: list[dict[str, Any]] = []
        self.demo_capture_frames = _env_flag("ROBENCH_DEMO_CAPTURE_FRAMES")
        self.demo_frame_count = 0

    def set_public_frame_dir(self, path: Path | str | None) -> None:
        self.public_frame_dir = Path(path) if path is not None else None

    def get_task_context(self) -> dict[str, Any]:
        context = public_task_context(self.task)
        context["benchmark_suite"] = self.suite
        context["scene"] = context.pop("floor_plan")
        context["max_note"] = "Use primitives only. Backend reset metadata, expert plans, low_actions, and masks are private."
        self._record_observation("get_task_context", {"result": context})
        return context

    def list_actions(self) -> list[str]:
        actions = [
            "observe()",
            "get_frame(label='frame')",
            "inspect_current_view(label=None, remember=True)",
            "detect_objects(query=None)",
            "remember_visible_objects(label=None, query=None)",
            "recall_visible_objects(query=None, label=None, limit=50)",
            "read_search_memory(query=None, label=None, limit=50)",
            "read_observed_spatial_map(query=None, limit=50)",
            "scan_scene(queries=None, rotations=4, include_tilts=True, remember=True)",
            "search_scene(queries=None, rounds=3, scan_rotations=4, include_tilts=True, remember=True)",
            "explore_room(queries=None, step_budget=32, include_tilts=True, remember=True)",
            "locate_object(query, aliases=None, support_queries=None, search_budget=36, include_tilts=True, open_containers=False)",
            "ground_object(query)",
            "query_object_state(obj)",
            "query_inventory()",
            "move_ahead()",
            "rotate(direction)",
            "look(direction)",
            "approach_object(obj, max_steps=3, stop_distance=1.25)",
            "open_object(obj)",
            "close_object(obj)",
            "pickup_object(obj)",
            "put_object(obj, receptacle)",
            "toggle_object(obj, on=True)",
            "slice_object(obj)",
            "pickup_located_object(query, aliases=None, support_queries=None, search_budget=36, open_containers=True)",
            "place_held_object(receptacle_query, obj=None, aliases=None, support_queries=None, search_budget=36)",
            "toggle_located_object(query, aliases=None, on=True, support_queries=None, search_budget=36)",
            "open_located_object(query, aliases=None, support_queries=None, search_budget=24)",
            "write_evidence(key, value)",
            "read_evidence()",
            "check_progress_public()",
            "check_success()",
        ]
        self._record_observation("list_actions", {"actions": actions})
        return actions

    def observe(self) -> dict[str, Any]:
        observation = self.backend.observe()
        self._update_observed_spatial_map(observation=observation)
        self._record_observation("observe", {"observation": observation})
        return observation

    def get_frame(self, label: str = "frame") -> dict[str, Any]:
        result = self._public_frame_result(self.backend.save_frame(self.task.get("task_id", "alfred_task"), str(label)))
        self._record_observation("get_frame", {"result": result})
        return result

    def inspect_current_view(self, label: str | None = None, remember: bool = True) -> dict[str, Any]:
        observation = self.backend.observe()
        visible = [_safe_object(obj) for obj in self.backend.visible_objects()]
        memory_label = label or "inspect_current_view"
        remembered = self._remember_objects(visible, label=memory_label, observation=observation) if remember else []
        frame = self._public_frame_result(
            self.backend.save_frame(self.task.get("task_id", "alfred_task"), memory_label)
        )
        result = {
            "label": memory_label,
            "remember": bool(remember),
            "frame": frame,
            "observation": observation,
            "visible_objects": visible,
            "remembered_count": len(remembered),
            "memory_count": len(self.visual_memory),
        }
        self._record_observation("inspect_current_view", {"result": result})
        return result

    def detect_objects(self, query: str | None = None, queries: Any = None) -> list[dict[str, Any]]:
        if query is None and queries is not None:
            query_list = _normalize_queries(queries)
            hits: list[dict[str, Any]] = []
            seen: set[str] = set()
            for item in query_list:
                for obj in self.detect_objects(item):
                    object_id = str(obj.get("objectId") or id(obj))
                    if object_id not in seen:
                        seen.add(object_id)
                        hits.append(obj)
            return hits
        objects = [_safe_object(obj) for obj in self.backend.visible_objects()]
        if query:
            objects = [obj for obj in objects if _matches(obj, query)]
        self._record_observation("detect_objects", {"query": query, "objects": objects})
        return objects

    def remember_visible_objects(self, label: str | None = None, query: str | None = None) -> dict[str, Any]:
        observation = self.backend.observe()
        objects = [_safe_object(obj) for obj in self.backend.visible_objects()]
        if query:
            objects = [obj for obj in objects if _matches(obj, query)]
        remembered = self._remember_objects(objects, label=label, observation=observation)
        result = {
            "label": label,
            "query": query,
            "remembered_count": len(remembered),
            "memory_count": len(self.visual_memory),
            "objects": remembered,
        }
        self._record_observation("remember_visible_objects", {"result": result})
        return result

    def recall_visible_objects(
        self,
        query: str | None = None,
        label: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        bounded_limit = max(1, min(int(limit), 200))
        objects = list(self.visual_memory.values())
        if query:
            objects = [obj for obj in objects if _matches(obj, query)]
        if label:
            norm_label = _norm(label)
            objects = [obj for obj in objects if _norm(obj.get("memory_label", "")) == norm_label]
        objects = sorted(objects, key=lambda obj: int(obj.get("last_seen_tick") or 0), reverse=True)
        result = [dict(obj) for obj in objects[:bounded_limit]]
        self._record_observation(
            "recall_visible_objects",
            {
                "query": query,
                "label": label,
                "limit": bounded_limit,
                "objects": result,
                "memory_count": len(self.visual_memory),
            },
        )
        return result

    def read_search_memory(
        self,
        query: str | None = None,
        label: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        bounded_limit = max(1, min(int(limit), 200))
        objects = list(self.visual_memory.values())
        if query:
            objects = [obj for obj in objects if _matches(obj, query)]
        if label:
            norm_label = _norm(label)
            objects = [obj for obj in objects if _norm(obj.get("memory_label", "")) == norm_label]
        objects = sorted(objects, key=lambda obj: int(obj.get("last_seen_tick") or 0), reverse=True)
        result = [dict(obj) for obj in objects[:bounded_limit]]
        self._record_observation(
            "read_search_memory",
            {
                "query": query,
                "label": label,
                "limit": bounded_limit,
                "objects": result,
                "memory_count": len(self.visual_memory),
            },
        )
        return result

    def read_observed_spatial_map(self, query: str | None = None, limit: int = 50) -> dict[str, Any]:
        observation = self.backend.observe()
        self._update_observed_spatial_map(observation=observation)
        bounded_limit = max(1, min(int(limit), 200))
        memories = list(self.visual_memory.values())
        if query:
            memories = [obj for obj in memories if _matches(obj, query)]
        memories = sorted(memories, key=lambda obj: int(obj.get("last_seen_tick") or 0), reverse=True)
        result = {
            "scene": observation.get("scene"),
            "current_agent": observation.get("agent"),
            "visited_cells": sorted(self.visited_cells.values(), key=lambda item: str(item.get("cell"))),
            "blocked_moves": list(self.blocked_moves[-bounded_limit:]),
            "frontiers": _frontiers_from_cells(self.visited_cells),
            "seen_objects": [dict(obj) for obj in memories[:bounded_limit]],
            "memory_count": len(self.visual_memory),
            "env_steps": self.metrics.get("env_steps", 0),
            "boundary": (
                "Observed-only spatial memory. It contains poses and objects seen through prior "
                "observations/actions; it does not expose hidden scene objects, target pose, full "
                "reachable map, shortest path, expert plan, masks, or low_actions."
            ),
        }
        self._record_observation(
            "read_observed_spatial_map",
            {"query": query, "limit": bounded_limit, "result": result},
        )
        return result

    def scan_scene(
        self,
        queries: Any = None,
        rotations: int = 4,
        include_tilts: bool = True,
        remember: bool = True,
    ) -> dict[str, Any]:
        query_list = _normalize_queries(queries)
        rotation_count = max(0, min(int(rotations), 8))
        observations: list[dict[str, Any]] = []
        hit_objects: dict[str, dict[str, dict[str, Any]]] = {query: {} for query in query_list}
        all_seen: dict[str, dict[str, Any]] = {}

        def capture(view_label: str) -> None:
            observation = self.backend.observe()
            visible = [_safe_object(obj) for obj in self.backend.visible_objects()]
            if remember:
                self._remember_objects(visible, label="scan_scene", observation=observation)
            for obj in visible:
                object_id = obj.get("objectId")
                if object_id:
                    all_seen[str(object_id)] = obj
            hits: dict[str, int] = {}
            for query in query_list:
                matches = [obj for obj in visible if _matches(obj, query)]
                hits[query] = len(matches)
                for obj in matches:
                    object_id = obj.get("objectId")
                    if object_id:
                        hit_objects[query][str(object_id)] = obj
            observations.append(
                {
                    "view": view_label,
                    "agent": observation.get("agent"),
                    "visible_count": len(visible),
                    "hits": hits,
                }
            )

        capture("initial")
        stopped_by_budget = False

        def can_step() -> bool:
            return self.active_max_env_steps is None or self.metrics["env_steps"] < self.active_max_env_steps

        def budgeted_step(step_fn: Any, *args: Any) -> bool:
            if not can_step():
                return False
            step_fn(*args)
            return True

        if include_tilts and can_step():
            if budgeted_step(self.look, "up"):
                capture("look_up")
            if budgeted_step(self.look, "down") and budgeted_step(self.look, "down"):
                capture("look_down")
            budgeted_step(self.look, "up")
        for index in range(rotation_count):
            if not budgeted_step(self.rotate, "left"):
                stopped_by_budget = True
                break
            capture(f"rotate_left_{index + 1}")
            if include_tilts:
                if not budgeted_step(self.look, "up"):
                    stopped_by_budget = True
                    break
                capture(f"rotate_left_{index + 1}_look_up")
                if not budgeted_step(self.look, "down"):
                    stopped_by_budget = True
                    break

        result = {
            "queries": query_list,
            "rotations": rotation_count,
            "include_tilts": bool(include_tilts),
            "remember": bool(remember),
            "observations": observations,
            "query_hits": {query: list(objects.values()) for query, objects in hit_objects.items()},
            "all_seen": list(all_seen.values()),
            "memory_count": len(self.visual_memory),
            "stopped_by_budget": stopped_by_budget or not can_step(),
        }
        self.trace.record_event(
            self.task,
            {
                "benchmark_suite": self.suite,
                "primitive": "scan_scene",
                "canonical_family": FAMILIES["scan_scene"],
                "side_effect": True,
                "result": result,
            },
        )
        self._mark_called("scan_scene", success=True)
        return result

    def search_scene(
        self,
        queries: Any = None,
        rounds: int = 3,
        scan_rotations: int = 4,
        include_tilts: bool = True,
        remember: bool = True,
        rotations: int | None = None,
    ) -> dict[str, Any]:
        if rotations is not None:
            scan_rotations = rotations
        query_list = _normalize_queries(queries)
        round_count = max(1, min(int(rounds), 6))
        aggregated_hits: dict[str, dict[str, dict[str, Any]]] = {query: {} for query in query_list}
        all_seen: dict[str, dict[str, Any]] = {}
        scans: list[dict[str, Any]] = []
        movement_events: list[dict[str, Any]] = []

        def can_step() -> bool:
            return self.active_max_env_steps is None or self.metrics["env_steps"] < self.active_max_env_steps

        for round_index in range(round_count):
            if not can_step():
                break
            scan = self.scan_scene(
                queries=query_list,
                rotations=scan_rotations,
                include_tilts=include_tilts,
                remember=remember,
            )
            scans.append(
                {
                    "round": round_index,
                    "views": len(scan.get("observations") or []),
                    "stopped_by_budget": bool(scan.get("stopped_by_budget")),
                    "hit_counts": {
                        query: len(objects)
                        for query, objects in (scan.get("query_hits") or {}).items()
                    },
                }
            )
            for query, objects in (scan.get("query_hits") or {}).items():
                bucket = aggregated_hits.setdefault(query, {})
                for obj in objects:
                    object_id = obj.get("objectId")
                    if object_id:
                        bucket[str(object_id)] = obj
            for obj in scan.get("all_seen") or []:
                object_id = obj.get("objectId")
                if object_id:
                    all_seen[str(object_id)] = obj
            if scan.get("stopped_by_budget") or not can_step():
                break

            move_result = self.move_ahead()
            movement_events.append(
                {
                    "round": round_index,
                    "primitive": "move_ahead",
                    "valid_action": move_result.valid_action,
                    "error": move_result.error,
                }
            )
            if not can_step():
                break
            if not move_result.valid_action:
                rotate_result = self.rotate("right")
                movement_events.append(
                    {
                        "round": round_index,
                        "primitive": "rotate",
                        "direction": "right",
                        "valid_action": rotate_result.valid_action,
                        "error": rotate_result.error,
                    }
                )

        result = {
            "queries": query_list,
            "rounds_requested": round_count,
            "rounds_completed": len(scans),
            "scan_rotations": max(0, min(int(scan_rotations), 8)),
            "include_tilts": bool(include_tilts),
            "remember": bool(remember),
            "scans": scans,
            "movement_events": movement_events,
            "query_hits": {query: list(objects.values()) for query, objects in aggregated_hits.items()},
            "all_seen": list(all_seen.values()),
            "memory_count": len(self.visual_memory),
            "stopped_by_budget": not can_step() or any(scan.get("stopped_by_budget") for scan in scans),
        }
        self.trace.record_event(
            self.task,
            {
                "benchmark_suite": self.suite,
                "primitive": "search_scene",
                "canonical_family": FAMILIES["search_scene"],
                "side_effect": True,
                "result": result,
            },
        )
        self._mark_called("search_scene", success=True)
        return result

    def explore_room(
        self,
        queries: Any = None,
        step_budget: int = 32,
        include_tilts: bool = True,
        remember: bool = True,
    ) -> dict[str, Any]:
        query_list = _normalize_queries(queries)
        action_limit = max(0, min(int(step_budget), 80))
        actions_used = 0
        segment_length = 1
        aggregated_hits: dict[str, dict[str, dict[str, Any]]] = {query: {} for query in query_list}
        all_seen: dict[str, dict[str, Any]] = {}
        observations: list[dict[str, Any]] = []
        movement_events: list[dict[str, Any]] = []

        def can_step() -> bool:
            return (
                actions_used < action_limit
                and (self.active_max_env_steps is None or self.metrics["env_steps"] < self.active_max_env_steps)
            )

        def capture(view_label: str) -> None:
            observation = self.backend.observe()
            visible = [_safe_object(obj) for obj in self.backend.visible_objects()]
            if remember:
                self._remember_objects(visible, label="explore_room", observation=observation)
            for obj in visible:
                object_id = obj.get("objectId")
                if object_id:
                    all_seen[str(object_id)] = obj
            hits: dict[str, int] = {}
            for query in query_list:
                matches = [obj for obj in visible if _matches(obj, query)]
                hits[query] = len(matches)
                bucket = aggregated_hits.setdefault(query, {})
                for obj in matches:
                    object_id = obj.get("objectId")
                    if object_id:
                        bucket[str(object_id)] = obj
            observations.append(
                {
                    "view": view_label,
                    "agent": observation.get("agent"),
                    "visible_count": len(visible),
                    "hits": hits,
                }
            )

        def run_action(label: str, primitive_fn: Any, *args: Any) -> StepResult | None:
            nonlocal actions_used
            if not can_step():
                return None
            before_steps = self.metrics["env_steps"]
            result = primitive_fn(*args)
            actions_used += max(1, self.metrics["env_steps"] - before_steps)
            movement_events.append(
                {
                    "action_index": actions_used,
                    "primitive": label,
                    "args": list(args),
                    "valid_action": result.valid_action,
                    "error": result.error,
                }
            )
            capture(f"{label}_{actions_used}")
            return result

        def tilt_sweep() -> None:
            if not include_tilts:
                return
            for direction in ("up", "down", "down", "up"):
                if not can_step():
                    break
                run_action("look", self.look, direction)

        capture("initial")
        tilt_sweep()
        while can_step():
            for _side in range(2):
                for _step in range(segment_length):
                    if not can_step():
                        break
                    move_result = run_action("move_ahead", self.move_ahead)
                    if move_result is not None and not move_result.valid_action:
                        if can_step():
                            run_action("rotate", self.rotate, "right")
                        retry_result = run_action("move_ahead", self.move_ahead) if can_step() else None
                        if retry_result is not None and not retry_result.valid_action:
                            if can_step():
                                run_action("rotate", self.rotate, "left")
                            if can_step():
                                run_action("rotate", self.rotate, "left")
                            run_action("move_ahead", self.move_ahead) if can_step() else None
                        break
                    if include_tilts and actions_used > 0 and actions_used % 12 == 0:
                        tilt_sweep()
                if not can_step():
                    break
                run_action("rotate", self.rotate, "left")
            segment_length = min(segment_length + 1, 6)

        result = {
            "queries": query_list,
            "step_budget": action_limit,
            "actions_used": actions_used,
            "include_tilts": bool(include_tilts),
            "remember": bool(remember),
            "observations": observations,
            "movement_events": movement_events,
            "query_hits": {query: list(objects.values()) for query, objects in aggregated_hits.items()},
            "all_seen": list(all_seen.values()),
            "memory_count": len(self.visual_memory),
            "stopped_by_budget": not can_step(),
        }
        self.trace.record_event(
            self.task,
            {
                "benchmark_suite": self.suite,
                "primitive": "explore_room",
                "canonical_family": FAMILIES["explore_room"],
                "side_effect": True,
                "result": result,
            },
        )
        self._mark_called("explore_room", success=True)
        return result

    def locate_object(
        self,
        query: Any,
        aliases: Any = None,
        support_queries: Any = None,
        search_budget: int = 36,
        include_tilts: bool = True,
        open_containers: bool = False,
    ) -> dict[str, Any]:
        target_queries = _unique_queries([query] + _normalize_queries(aliases))
        support_query_list = _unique_queries(_normalize_queries(support_queries))
        search_queries = _unique_queries(target_queries + support_query_list)
        step_budget = _bounded_int(search_budget, default=36, low=0, high=80)
        arguments = {
            "query": query,
            "aliases": aliases,
            "support_queries": support_queries,
            "search_budget": step_budget,
            "include_tilts": bool(include_tilts),
            "open_containers": bool(open_containers),
        }
        result: dict[str, Any] = {
            "query": query,
            "target_queries": target_queries,
            "support_queries": support_query_list,
            "search_budget": step_budget,
            "found": False,
            "visible_now": False,
            "current_object": None,
            "best_candidate": None,
            "candidates": [],
            "events": [],
            "boundary": (
                "Observed-only bounded locate skill. It searches through visible frames and run-local "
                "visual memory; it does not use expert actions, hidden object ids, masks, shortest paths, "
                "or a simulator scene graph."
            ),
        }
        if not target_queries:
            result["error"] = {"kind": "empty_query", "message": "locate_object requires a non-empty query."}
            self._record_composite("locate_object", arguments, result, success=False)
            return result

        current = self._first_visible_match(target_queries)
        if current is not None:
            result.update(
                {
                    "found": True,
                    "visible_now": True,
                    "current_object": current,
                    "best_candidate": current,
                    "candidates": [current],
                }
            )
            self._record_composite("locate_object", arguments, result, success=True)
            return result

        if step_budget > 0:
            scan = self.search_scene(
                queries=search_queries,
                rounds=1,
                scan_rotations=4,
                include_tilts=include_tilts,
                remember=True,
            )
            result["events"].append(
                {
                    "primitive": "search_scene",
                    "hit_counts": _query_hit_counts(scan.get("query_hits") or {}),
                    "stopped_by_budget": bool(scan.get("stopped_by_budget")),
                }
            )
            current = self._first_visible_match(target_queries)

        if current is None and open_containers and step_budget > 0:
            opened = self._open_visible_containers(support_query_list, limit=2)
            if opened:
                result["events"].append({"primitive": "open_visible_containers", "opened": opened})
                scan = self.scan_scene(
                    queries=target_queries,
                    rotations=2,
                    include_tilts=include_tilts,
                    remember=True,
                )
                result["events"].append(
                    {
                        "primitive": "scan_scene_after_open",
                        "hit_counts": _query_hit_counts(scan.get("query_hits") or {}),
                        "stopped_by_budget": bool(scan.get("stopped_by_budget")),
                    }
                )
                current = self._first_visible_match(target_queries)

        candidate = current or self._first_memory_match(target_queries)
        if candidate is None and support_query_list:
            support = self._first_visible_match(support_query_list) or self._first_memory_match(support_query_list)
            if support is not None:
                approach = self.approach_object(support, max_steps=3)
                result["events"].append(
                    {
                        "primitive": "approach_support",
                        "target": _object_brief(support),
                        "result": _approach_result_summary(approach),
                    }
                )
                scan = self._scan_until_visible(
                    target_queries,
                    rotations=4,
                    include_tilts=include_tilts,
                    remember=True,
                    label="scan_scene_after_support",
                )
                result["events"].append(
                    {
                        "primitive": "scan_scene_after_support",
                        "hit_counts": _query_hit_counts(scan.get("query_hits") or {}),
                        "stopped_by_budget": bool(scan.get("stopped_by_budget")),
                    }
                )
                current = self._first_visible_match(target_queries)
                candidate = current or self._first_memory_match(target_queries)

        if candidate is None and step_budget > 8:
            explore_steps = max(8, min(step_budget, 48))
            explore = self.explore_room(
                queries=search_queries,
                step_budget=explore_steps,
                include_tilts=include_tilts,
                remember=True,
            )
            result["events"].append(
                {
                    "primitive": "explore_room",
                    "actions_used": explore.get("actions_used"),
                    "hit_counts": _query_hit_counts(explore.get("query_hits") or {}),
                    "stopped_by_budget": bool(explore.get("stopped_by_budget")),
                }
            )
            current = self._first_visible_match(target_queries)
            candidate = current or self._first_memory_match(target_queries)

        if candidate is not None and current is None:
            approach_steps = max(2, min(5, step_budget // 8 if step_budget else 3))
            approach = self.approach_object(candidate, max_steps=approach_steps)
            result["events"].append(
                {
                    "primitive": "approach_candidate",
                    "target": _object_brief(candidate),
                    "result": _approach_result_summary(approach),
                }
            )
            scan = self._scan_until_visible(
                target_queries,
                rotations=4,
                include_tilts=include_tilts,
                remember=True,
                label="reacquire_after_approach",
            )
            result["events"].append(
                {
                    "primitive": "reacquire_after_approach",
                    "hit_counts": _query_hit_counts(scan.get("query_hits") or {}),
                    "stopped_by_budget": bool(scan.get("stopped_by_budget")),
                }
            )
            current = self._first_visible_match(target_queries)

        if candidate is not None and current is None:
            approach = self.approach_object(candidate, max_steps=2, stop_distance=1.0)
            result["events"].append(
                {
                    "primitive": "final_reacquire_approach",
                    "target": _object_brief(candidate),
                    "result": _approach_result_summary(approach),
                }
            )
            scan = self._scan_until_visible(
                target_queries,
                rotations=4,
                include_tilts=include_tilts,
                remember=True,
                label="final_reacquire_scan",
            )
            result["events"].append(
                {
                    "primitive": "final_reacquire_scan",
                    "hit_counts": _query_hit_counts(scan.get("query_hits") or {}),
                    "stopped_by_budget": bool(scan.get("stopped_by_budget")),
                }
            )
            current = self._first_visible_match(target_queries)

        memories = self._memory_matches(target_queries, limit=8)
        candidates = _dedupe_objects(([current] if current else []) + ([candidate] if candidate else []) + memories)
        result.update(
            {
                "found": bool(current or candidate or candidates),
                "visible_now": current is not None,
                "current_object": current,
                "best_candidate": current or candidate or (candidates[0] if candidates else None),
                "candidates": candidates,
                "memory_count": len(self.visual_memory),
                "spatial_memory": self.read_observed_spatial_map(query=target_queries[0], limit=20),
            }
        )
        self._record_composite("locate_object", arguments, result, success=bool(result["found"]))
        return result

    def pickup_located_object(
        self,
        query: Any,
        aliases: Any = None,
        support_queries: Any = None,
        search_budget: int = 36,
        open_containers: bool = True,
    ) -> dict[str, Any]:
        arguments = {
            "query": query,
            "aliases": aliases,
            "support_queries": support_queries,
            "search_budget": search_budget,
            "open_containers": bool(open_containers),
        }
        located = self.locate_object(
            query,
            aliases=aliases,
            support_queries=support_queries,
            search_budget=search_budget,
            include_tilts=True,
            open_containers=open_containers,
        )
        target = located.get("current_object") if isinstance(located, dict) else None
        if not target:
            result = {
                "ok": False,
                "stage": "locate",
                "locate": _locate_result_summary(located),
                "error": {"kind": "not_visible_after_locate", "message": "Target was not visible after bounded locate."},
            }
            self._record_composite("pickup_located_object", arguments, result, success=False)
            return result
        pickup = self.pickup_object(target["objectId"])
        result = {
            "ok": bool(pickup.valid_action),
            "stage": "pickup",
            "target": _object_brief(target),
            "locate": _locate_result_summary(located),
            "pickup": _step_result_summary(pickup),
            "inventory": self.query_inventory(),
            "progress": self.check_progress_public(),
        }
        self._record_composite("pickup_located_object", arguments, result, success=bool(pickup.valid_action))
        return result

    def place_held_object(
        self,
        receptacle_query: Any,
        obj: Any = None,
        aliases: Any = None,
        support_queries: Any = None,
        search_budget: int = 36,
    ) -> dict[str, Any]:
        arguments = {
            "receptacle_query": receptacle_query,
            "obj": obj,
            "aliases": aliases,
            "support_queries": support_queries,
            "search_budget": search_budget,
        }
        inventory = self.query_inventory()
        if not inventory:
            result = {
                "ok": False,
                "stage": "inventory",
                "error": {"kind": "empty_inventory", "message": "place_held_object requires a held object."},
            }
            self._record_composite("place_held_object", arguments, result, success=False)
            return result
        held_ref = obj or inventory[0].get("objectId")
        receptacle_queries = _unique_queries([receptacle_query] + _normalize_queries(aliases))
        located = self.locate_object(
            receptacle_query,
            aliases=aliases,
            support_queries=support_queries,
            search_budget=search_budget,
            include_tilts=True,
            open_containers=False,
        )
        receptacle = located.get("current_object") if isinstance(located, dict) else None
        if not receptacle:
            result = {
                "ok": False,
                "stage": "locate_receptacle",
                "held_object": _object_brief(inventory[0]),
                "locate": _locate_result_summary(located),
                "error": {
                    "kind": "receptacle_not_visible_after_locate",
                    "message": "Receptacle was not visible after bounded locate.",
                },
            }
            self._record_composite("place_held_object", arguments, result, success=False)
            return result
        put = self.put_object(held_ref, receptacle["objectId"])
        put_attempts = [{"receptacle": _object_brief(receptacle), "put": _step_result_summary(put)}]
        retry_event: dict[str, Any] | None = None
        if not put.valid_action:
            approach_retry = self.approach_object(receptacle, max_steps=3, stop_distance=0.75)
            reacquire = self._scan_until_visible(
                receptacle_queries,
                rotations=2,
                include_tilts=True,
                remember=True,
                label="place_retry_reacquire",
            )
            retry_receptacle = self._first_visible_match(receptacle_queries)
            retry_event = {
                "approach": _approach_result_summary(approach_retry),
                "reacquire_hit_counts": _query_hit_counts(reacquire.get("query_hits") or {}),
                "retry_receptacle": _object_brief(retry_receptacle),
            }
            if retry_receptacle is not None:
                retry_put = self.put_object(held_ref, retry_receptacle["objectId"])
                put_attempts.append({"receptacle": _object_brief(retry_receptacle), "put": _step_result_summary(retry_put)})
                if retry_put.valid_action:
                    receptacle = retry_receptacle
                    put = retry_put
        result = {
            "ok": bool(put.valid_action),
            "stage": "put",
            "held_object": _object_brief(inventory[0]),
            "receptacle": _object_brief(receptacle),
            "locate": _locate_result_summary(located),
            "put": _step_result_summary(put),
            "put_attempts": put_attempts,
            "retry": retry_event,
            "progress": self.check_progress_public(),
        }
        self._record_composite("place_held_object", arguments, result, success=bool(put.valid_action))
        return result

    def toggle_located_object(
        self,
        query: Any,
        aliases: Any = None,
        on: bool = True,
        support_queries: Any = None,
        search_budget: int = 36,
    ) -> dict[str, Any]:
        arguments = {
            "query": query,
            "aliases": aliases,
            "on": bool(on),
            "support_queries": support_queries,
            "search_budget": search_budget,
        }
        located = self.locate_object(
            query,
            aliases=aliases,
            support_queries=support_queries,
            search_budget=search_budget,
            include_tilts=True,
            open_containers=False,
        )
        target = located.get("current_object") if isinstance(located, dict) else None
        if not target:
            result = {
                "ok": False,
                "stage": "locate",
                "locate": _locate_result_summary(located),
                "error": {"kind": "not_visible_after_locate", "message": "Toggle target was not visible after bounded locate."},
            }
            self._record_composite("toggle_located_object", arguments, result, success=False)
            return result
        toggled = self.toggle_object(target["objectId"], on=on)
        result = {
            "ok": bool(toggled.valid_action),
            "stage": "toggle",
            "target": _object_brief(target),
            "locate": _locate_result_summary(located),
            "toggle": _step_result_summary(toggled),
            "progress": self.check_progress_public(),
        }
        self._record_composite("toggle_located_object", arguments, result, success=bool(toggled.valid_action))
        return result

    def open_located_object(
        self,
        query: Any,
        aliases: Any = None,
        support_queries: Any = None,
        search_budget: int = 24,
    ) -> dict[str, Any]:
        arguments = {
            "query": query,
            "aliases": aliases,
            "support_queries": support_queries,
            "search_budget": search_budget,
        }
        located = self.locate_object(
            query,
            aliases=aliases,
            support_queries=support_queries,
            search_budget=search_budget,
            include_tilts=True,
            open_containers=False,
        )
        target = located.get("current_object") if isinstance(located, dict) else None
        if not target:
            result = {
                "ok": False,
                "stage": "locate",
                "locate": _locate_result_summary(located),
                "error": {"kind": "not_visible_after_locate", "message": "Open target was not visible after bounded locate."},
            }
            self._record_composite("open_located_object", arguments, result, success=False)
            return result
        opened = self.open_object(target["objectId"])
        result = {
            "ok": bool(opened.valid_action),
            "stage": "open",
            "target": _object_brief(target),
            "locate": _locate_result_summary(located),
            "open": _step_result_summary(opened),
            "progress": self.check_progress_public(),
        }
        self._record_composite("open_located_object", arguments, result, success=bool(opened.valid_action))
        return result

    def ground_object(self, query: str) -> dict[str, Any]:
        resolved = self._resolve_object(query, visible_only=True)
        if "error" in resolved:
            self._record_observation("ground_object", {"query": query, "error": resolved})
            return resolved
        result = dict(resolved["object"])
        self._record_observation("ground_object", {"query": query, "result": result})
        return result

    def query_object_state(self, obj: Any) -> dict[str, Any]:
        resolved = self._resolve_object(obj, visible_only=False)
        if "error" in resolved:
            self._record_observation("query_object_state", {"obj": obj, "error": resolved})
            return resolved
        result = dict(resolved["object"])
        self._record_observation("query_object_state", {"obj": obj, "result": result})
        return result

    def query_inventory(self) -> list[dict[str, Any]]:
        inventory = [_safe_object(obj) for obj in self.backend.inventory_objects()]
        self._record_observation("query_inventory", {"inventory": inventory})
        return inventory

    def move_ahead(self) -> StepResult:
        return self._execute("move_ahead", {"action": "MoveAhead", "forceAction": True}, {})

    def rotate(self, direction: str) -> StepResult:
        action = ROTATE_ACTIONS.get(_norm(direction))
        if action is None:
            return self._invalid("rotate", {"direction": direction}, "invalid_direction", "Use left or right.")
        return self._execute("rotate", {"action": action, "forceAction": True}, {"direction": direction})

    def look(self, direction: str) -> StepResult:
        action = LOOK_ACTIONS.get(_norm(direction))
        if action is None:
            return self._invalid("look", {"direction": direction}, "invalid_direction", "Use up or down.")
        return self._execute("look", {"action": action, "forceAction": True}, {"direction": direction})

    def approach_object(self, obj: Any, max_steps: int = 3, stop_distance: float = 1.25) -> dict[str, Any]:
        resolved = self._resolve_visible_or_memory_object(obj)
        arguments = {"obj": obj, "max_steps": max_steps, "stop_distance": stop_distance}
        if "error" in resolved:
            result = self._invalid(
                "approach_object",
                arguments,
                resolved["error"],
                resolved["message"],
                resolved.get("candidates"),
            )
            return {
                "valid_action": False,
                "error": result.error,
                "events": [],
                "final_observation": result.observation_after,
            }
        target = resolved["object"]
        target_position = target.get("position")
        if not isinstance(target_position, dict):
            result = self._invalid(
                "approach_object",
                arguments,
                "no_position",
                "Object has no remembered/current position.",
                [target],
            )
            return {
                "valid_action": False,
                "error": result.error,
                "events": [],
                "final_observation": result.observation_after,
            }

        step_limit = max(0, min(int(max_steps), 10))
        distance_limit = max(0.25, float(stop_distance))
        events: list[dict[str, Any]] = []

        def can_step() -> bool:
            return self.active_max_env_steps is None or self.metrics["env_steps"] < self.active_max_env_steps

        for _index in range(step_limit):
            observation = self.backend.observe()
            agent_position = (observation.get("agent") or {}).get("position") or {}
            agent_rotation = ((observation.get("agent") or {}).get("rotation") or {}).get("y", 0.0)
            distance = _planar_distance(agent_position, target_position)
            if distance is not None and distance <= distance_limit:
                break
            if not can_step():
                break

            direction = _coarse_turn_direction(agent_position, target_position, agent_rotation)
            if direction:
                step = self.rotate(direction)
                events.append(
                    {
                        "primitive": "rotate",
                        "direction": direction,
                        "valid_action": step.valid_action,
                        "error": step.error,
                    }
                )
                if not step.valid_action:
                    break
                continue

            step = self.move_ahead()
            events.append(
                {
                    "primitive": "move_ahead",
                    "valid_action": step.valid_action,
                    "error": step.error,
                }
            )
            if not step.valid_action and can_step():
                rotate_step = self.rotate("right")
                events.append(
                    {
                        "primitive": "rotate",
                        "direction": "right",
                        "valid_action": rotate_step.valid_action,
                        "error": rotate_step.error,
                    }
                )

        final_observation = self.backend.observe()
        final_position = (final_observation.get("agent") or {}).get("position") or {}
        result = {
            "valid_action": True,
            "target": target,
            "events": events,
            "final_distance": _planar_distance(final_position, target_position),
            "final_observation": final_observation,
            "stopped_by_budget": not can_step(),
        }
        self.trace.record_event(
            self.task,
            {
                "benchmark_suite": self.suite,
                "primitive": "approach_object",
                "canonical_family": FAMILIES["approach_object"],
                "side_effect": True,
                "arguments": arguments,
                "result": result,
                "evidence": dict(self.evidence),
                "provenance": dict(self.provenance),
            },
        )
        self._mark_called("approach_object", success=True)
        return result

    def open_object(self, obj: Any) -> StepResult:
        return self._object_action("open_object", "OpenObject", obj)

    def close_object(self, obj: Any) -> StepResult:
        return self._object_action("close_object", "CloseObject", obj, {"forceAction": True})

    def pickup_object(self, obj: Any) -> StepResult:
        return self._object_action("pickup_object", "PickupObject", obj)

    def put_object(self, obj: Any, receptacle: Any) -> StepResult:
        held = self._resolve_inventory_object(obj)
        if "error" in held:
            return self._invalid("put_object", {"obj": obj, "receptacle": receptacle}, held["error"], held["message"], held.get("candidates"))
        target = self._resolve_object(receptacle, visible_only=True)
        if "error" in target:
            return self._invalid(
                "put_object",
                {"obj": obj, "receptacle": receptacle},
                target["error"],
                target["message"],
                target.get("candidates"),
            )
        return self._execute(
            "put_object",
            {
                "action": "PutObject",
                "objectId": held["object"]["objectId"],
                "receptacleObjectId": target["object"]["objectId"],
                "forceAction": True,
                "placeStationary": True,
            },
            {
                "obj": obj,
                "held_objectId": held["object"]["objectId"],
                "held_objectType": held["object"].get("objectType"),
                "receptacle": receptacle,
                "receptacle_objectId": target["object"]["objectId"],
                "receptacle_objectType": target["object"].get("objectType"),
            },
        )

    def toggle_object(self, obj: Any, on: bool = True) -> StepResult:
        action = "ToggleObjectOn" if bool(on) else "ToggleObjectOff"
        return self._object_action("toggle_object", action, obj, {"on": bool(on)})

    def slice_object(self, obj: Any) -> StepResult:
        return self._object_action("slice_object", "SliceObject", obj)

    def write_evidence(self, key: str, value: Any) -> dict[str, Any]:
        self.evidence[str(key)] = value
        result = {"key": str(key), "value": value}
        self._record_observation("write_evidence", {"result": result, "evidence": dict(self.evidence)})
        return result

    def read_evidence(self) -> dict[str, Any]:
        evidence = dict(self.evidence)
        self._record_observation("read_evidence", {"evidence": evidence})
        return evidence

    def check_success(self) -> dict[str, Any]:
        result = self.backend.check_success()
        self._record_observation("check_success", {"verification": result, "evidence": dict(self.evidence)})
        return result

    def check_progress_public(self) -> dict[str, Any]:
        verification = self.backend.check_success()
        observation = self.backend.observe()
        result = {
            "success": bool(verification.get("success")),
            "completed": bool(verification.get("completed")),
            "score": verification.get("score"),
            "goal_conditions_met": verification.get("goal_conditions_met"),
            "goal_conditions_total": verification.get("goal_conditions_total"),
            "env_steps": self.metrics.get("env_steps", 0),
            "visible_object_count": len(observation.get("visible_objects") or []),
            "inventory": observation.get("inventory") or [],
            "memory_count": len(self.visual_memory),
            "note": "Public verifier summary only; no expert low_actions, masks, or hidden target ids.",
        }
        self._record_observation("check_progress_public", {"result": result})
        return result

    def record_harness_event(self, event_name: str, payload: dict[str, Any], side_effect: bool = False) -> None:
        self.trace.record_event(
            self.task,
            {
                "benchmark_suite": self.suite,
                "primitive": event_name,
                "canonical_family": FAMILIES[event_name],
                "side_effect": side_effect,
                **payload,
            },
        )

    def _object_action(
        self,
        primitive: str,
        action: str,
        obj: Any,
        extra_native: dict[str, Any] | None = None,
    ) -> StepResult:
        resolved = self._resolve_object(obj, visible_only=True)
        arguments = {"obj": obj}
        if "error" in resolved:
            return self._invalid(primitive, arguments, resolved["error"], resolved["message"], resolved.get("candidates"))
        object_id = resolved["object"]["objectId"]
        arguments = {**arguments, "objectId": object_id, "objectType": resolved["object"].get("objectType")}
        native = {"action": action, "objectId": object_id}
        native.update(extra_native or {})
        return self._execute(primitive, native, arguments)

    def _execute(self, primitive: str, native_action: dict[str, Any], arguments: dict[str, Any]) -> StepResult:
        result = self.backend.step(primitive, native_action)
        self.metrics["env_steps"] += 1
        if not result.valid_action:
            self.metrics["invalid_action_count"] += 1
        self._update_observed_spatial_map(
            observation=result.observation_after,
            primitive=primitive,
            valid_action=result.valid_action,
            error=result.error,
        )
        self._record_step(primitive, arguments, result)
        return result

    def _invalid(
        self,
        primitive: str,
        arguments: dict[str, Any],
        kind: str,
        message: str,
        candidates: Any = None,
    ) -> StepResult:
        if kind == "ambiguous":
            self.metrics["wrapper_ambiguity_count"] += 1
        elif kind == "no_match":
            self.metrics["wrapper_no_match_count"] += 1
        error = {"kind": kind, "message": message, "candidates": candidates}
        result = self.backend.invalid_result(primitive, error)
        self._update_observed_spatial_map(observation=result.observation_after)
        self._record_step(primitive, arguments, result)
        return result

    def _resolve_inventory_object(self, ref: Any) -> dict[str, Any]:
        return _resolve_from_objects(ref, self.backend.inventory_objects())

    def _resolve_object(self, ref: Any, visible_only: bool) -> dict[str, Any]:
        objects = self.backend.visible_objects() if visible_only else self.backend.visible_objects() + self.backend.inventory_objects()
        return _resolve_from_objects(ref, objects)

    def _resolve_visible_or_memory_object(self, ref: Any) -> dict[str, Any]:
        visible = self._resolve_object(ref, visible_only=True)
        if "error" not in visible:
            return visible
        memory = _resolve_from_objects(ref, list(self.visual_memory.values()))
        if "error" not in memory:
            return memory
        return visible

    def _first_visible_match(self, queries: list[str]) -> dict[str, Any] | None:
        visible = [_safe_object(obj) for obj in self.backend.visible_objects()]
        for query in queries:
            for obj in visible:
                if _matches(obj, query):
                    return obj
        return None

    def _memory_matches(self, queries: list[str], limit: int = 8) -> list[dict[str, Any]]:
        memories = list(self.visual_memory.values())
        matches: list[dict[str, Any]] = []
        for query in queries:
            matches.extend(obj for obj in memories if _matches(obj, query))
        matches = sorted(matches, key=lambda obj: int(obj.get("last_seen_tick") or 0), reverse=True)
        return _dedupe_objects(matches)[: max(1, int(limit))]

    def _first_memory_match(self, queries: list[str]) -> dict[str, Any] | None:
        matches = self._memory_matches(queries, limit=1)
        return matches[0] if matches else None

    def _open_visible_containers(self, container_queries: list[str], limit: int = 2) -> list[dict[str, Any]]:
        visible = [_safe_object(obj) for obj in self.backend.visible_objects()]
        if container_queries:
            visible = [obj for obj in visible if any(_matches(obj, query) for query in container_queries)]
        openable = [
            obj
            for obj in visible
            if bool(obj.get("openable")) and not bool(obj.get("isOpen")) and obj.get("objectId")
        ]
        opened: list[dict[str, Any]] = []
        for obj in openable[: max(0, min(int(limit), 4))]:
            step = self.open_object(obj["objectId"])
            opened.append({"object": _object_brief(obj), "result": _step_result_summary(step)})
            if self.active_max_env_steps is not None and self.metrics["env_steps"] >= self.active_max_env_steps:
                break
        return opened

    def _scan_until_visible(
        self,
        queries: list[str],
        *,
        rotations: int,
        include_tilts: bool,
        remember: bool,
        label: str,
    ) -> dict[str, Any]:
        query_list = _unique_queries(queries)
        rotation_count = max(0, min(int(rotations), 8))
        observations: list[dict[str, Any]] = []
        hit_objects: dict[str, dict[str, dict[str, Any]]] = {query: {} for query in query_list}
        all_seen: dict[str, dict[str, Any]] = {}

        def can_step() -> bool:
            return self.active_max_env_steps is None or self.metrics["env_steps"] < self.active_max_env_steps

        def capture(view_label: str) -> dict[str, Any] | None:
            observation = self.backend.observe()
            visible = [_safe_object(obj) for obj in self.backend.visible_objects()]
            if remember:
                self._remember_objects(visible, label=label, observation=observation)
            for obj in visible:
                object_id = obj.get("objectId")
                if object_id:
                    all_seen[str(object_id)] = obj
            hits: dict[str, int] = {}
            first_match: dict[str, Any] | None = None
            for query in query_list:
                matches = [obj for obj in visible if _matches(obj, query)]
                hits[query] = len(matches)
                for obj in matches:
                    object_id = obj.get("objectId")
                    if object_id:
                        hit_objects[query][str(object_id)] = obj
                    if first_match is None:
                        first_match = obj
            observations.append(
                {
                    "view": view_label,
                    "agent": observation.get("agent"),
                    "visible_count": len(visible),
                    "hits": hits,
                }
            )
            return first_match

        found = capture("initial")
        if found is None and include_tilts:
            for direction in ("up", "down", "down", "up"):
                if not can_step():
                    break
                self.look(direction)
                found = capture(f"look_{direction}")
                if found is not None:
                    break
        rotation_index = 0
        while found is None and rotation_index < rotation_count and can_step():
            rotation_index += 1
            self.rotate("left")
            found = capture(f"rotate_left_{rotation_index}")
            if found is not None:
                break
            if include_tilts:
                for direction in ("up", "down", "down", "up"):
                    if not can_step():
                        break
                    self.look(direction)
                    found = capture(f"rotate_left_{rotation_index}_look_{direction}")
                    if found is not None:
                        break

        return {
            "queries": query_list,
            "rotations": rotation_count,
            "include_tilts": bool(include_tilts),
            "remember": bool(remember),
            "observations": observations,
            "query_hits": {query: list(objects.values()) for query, objects in hit_objects.items()},
            "all_seen": list(all_seen.values()),
            "memory_count": len(self.visual_memory),
            "found": found,
            "stopped_on_hit": found is not None,
            "stopped_by_budget": found is None and not can_step(),
        }

    def _remember_objects(
        self,
        objects: list[dict[str, Any]],
        *,
        label: str | None,
        observation: dict[str, Any],
    ) -> list[dict[str, Any]]:
        remembered: list[dict[str, Any]] = []
        for obj in objects:
            object_id = obj.get("objectId")
            if not object_id:
                continue
            self.memory_tick += 1
            previous = self.visual_memory.get(str(object_id), {})
            entry = {
                **obj,
                "memory_label": label,
                "first_seen_tick": previous.get("first_seen_tick", self.memory_tick),
                "last_seen_tick": self.memory_tick,
                "seen_count": int(previous.get("seen_count", 0) or 0) + 1,
                "last_seen_agent": observation.get("agent"),
                "last_seen_scene": observation.get("scene"),
            }
            self.visual_memory[str(object_id)] = entry
            remembered.append(dict(entry))
        return remembered

    def _update_observed_spatial_map(
        self,
        *,
        observation: dict[str, Any] | None,
        primitive: str | None = None,
        valid_action: bool | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        if not observation:
            return
        agent = observation.get("agent") or {}
        position = agent.get("position") or {}
        cell = _pose_cell(position)
        if cell:
            self.visited_cells[cell] = {
                "cell": cell,
                "position": _round_position(position),
                "rotation_y": _round(((agent.get("rotation") or {}).get("y"))),
                "cameraHorizon": _round(agent.get("cameraHorizon")),
                "last_seen_step": self.metrics.get("env_steps", 0),
                "visit_count": int((self.visited_cells.get(cell) or {}).get("visit_count", 0) or 0) + 1,
            }
        objects = [_safe_object(obj) for obj in observation.get("visible_objects") or []]
        if objects:
            self._remember_objects(objects, label="observed_spatial_map", observation=observation)
        if valid_action is False and primitive in {"move_ahead", "rotate", "look", "approach_object"}:
            self.blocked_moves.append(
                {
                    "step": self.metrics.get("env_steps", 0),
                    "primitive": primitive,
                    "agent": agent,
                    "error": error,
                }
            )
            self.blocked_moves = self.blocked_moves[-100:]

    def _public_frame_result(self, result: dict[str, Any]) -> dict[str, Any]:
        public = dict(result)
        source_path = public.get("path")
        if not public.get("saved") or not source_path:
            public["path"] = None
            return public
        source = Path(str(source_path))
        if self.public_frame_dir is not None and source.exists():
            self.public_frame_dir.mkdir(parents=True, exist_ok=True)
            dest = self.public_frame_dir / source.name
            shutil.copy2(source, dest)
            public["path"] = str(Path("frames") / dest.name)
            public["path_boundary"] = "workspace-relative"
            return public
        public["path"] = source.name
        public["path_boundary"] = "filename-only; backend absolute path hidden"
        return public

    def _record_observation(self, primitive: str, payload: dict[str, Any]) -> None:
        self._mark_called(primitive)
        payload = dict(payload)
        demo_frame = self._capture_demo_frame(primitive)
        if demo_frame is not None:
            payload["demo_frame"] = demo_frame
        self.trace.record_event(
            self.task,
            {
                "benchmark_suite": self.suite,
                "primitive": primitive,
                "canonical_family": FAMILIES[primitive],
                "side_effect": False,
                **payload,
            },
        )

    def _record_step(self, primitive: str, arguments: dict[str, Any], result: StepResult) -> None:
        self._mark_called(primitive, success=result.valid_action, arguments=arguments)
        payload: dict[str, Any] = {
            "benchmark_suite": self.suite,
            "primitive": primitive,
            "canonical_family": FAMILIES[primitive],
            "side_effect": True,
            "arguments": arguments,
            "native_action": result.native_action,
            "valid_action": result.valid_action,
            "observation_before": result.observation_before,
            "observation_after": result.observation_after,
            "verifier_result": result.verification,
            "success": result.success,
            "completed": result.completed,
            "score": result.score,
            "error": result.error,
            "evidence": dict(self.evidence),
            "provenance": dict(self.provenance),
        }
        demo_frame = self._capture_demo_frame(primitive)
        if demo_frame is not None:
            payload["demo_frame"] = demo_frame
        self.trace.record_event(
            self.task,
            payload,
        )

    def _record_composite(
        self,
        primitive: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
        *,
        success: bool,
    ) -> None:
        self._mark_called(primitive, success=success, arguments=arguments)
        payload: dict[str, Any] = {
            "benchmark_suite": self.suite,
            "primitive": primitive,
            "canonical_family": FAMILIES[primitive],
            "side_effect": True,
            "arguments": arguments,
            "result": result,
            "success": bool(success),
            "evidence": dict(self.evidence),
            "provenance": dict(self.provenance),
        }
        demo_frame = self._capture_demo_frame(primitive)
        if demo_frame is not None:
            payload["demo_frame"] = demo_frame
        self.trace.record_event(self.task, payload)

    def _capture_demo_frame(self, primitive: str) -> dict[str, Any] | None:
        if not self.demo_capture_frames or primitive == "get_frame":
            return None
        self.demo_frame_count += 1
        label = f"demo_{self.demo_frame_count:04d}_{_safe_frame_label(primitive)}"
        result = self._public_frame_result(self.backend.save_frame(self.task.get("task_id", "alfred_task"), label))
        if not result.get("saved"):
            return None
        result["capture_mode"] = "ROBENCH_DEMO_CAPTURE_FRAMES"
        return result

    def _mark_called(
        self,
        primitive: str,
        success: bool | None = None,
        arguments: dict[str, Any] | None = None,
    ) -> None:
        count = int(self.provenance.get(primitive, 0) or 0)
        self.provenance[primitive] = count + 1
        event: dict[str, Any] = {"primitive": primitive}
        if success is not None:
            event["success"] = bool(success)
            if success:
                success_key = f"{primitive}:success"
                self.provenance[success_key] = int(self.provenance.get(success_key, 0) or 0) + 1
        if arguments:
            for key in ("objectId", "objectType", "held_objectId", "held_objectType", "receptacle_objectId", "receptacle_objectType"):
                if key in arguments:
                    event[key] = arguments[key]
        events = self.provenance.setdefault("_events", [])
        if isinstance(events, list):
            events.append(event)


def _resolve_from_objects(ref: Any, objects: list[dict[str, Any]]) -> dict[str, Any]:
    safe_objects = [_safe_object(obj) for obj in objects]
    if isinstance(ref, dict) and ref.get("objectId"):
        ref = ref["objectId"]
    query = str(ref).strip()
    object_id_matches = [obj for obj in safe_objects if obj.get("objectId") == query]
    if len(object_id_matches) == 1:
        return {"object": object_id_matches[0]}

    exact = [obj for obj in safe_objects if _norm(obj.get("objectType", "")) == _norm(query) or _norm(obj.get("name", "")) == _norm(query)]
    candidates = exact or [obj for obj in safe_objects if _matches(obj, query)]
    unique = {obj.get("objectId"): obj for obj in candidates if obj.get("objectId")}
    if len(unique) == 1:
        return {"object": next(iter(unique.values()))}
    if len(unique) > 1:
        return {"error": "ambiguous", "message": f"Multiple visible/current objects matched {ref!r}. Use objectId.", "candidates": list(unique.values())}
    return {"error": "no_match", "message": f"No visible/current object matched {ref!r}.", "candidates": safe_objects}


def _unique_queries(queries: list[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for query in queries:
        text = str(query).strip()
        if not text:
            continue
        key = _norm(text)
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _bounded_int(value: Any, *, default: int, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(number, high))


def _dedupe_objects(objects: list[dict[str, Any] | None]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for obj in objects:
        if not obj:
            continue
        object_id = str(obj.get("objectId") or "")
        key = object_id or repr(sorted(obj.items()))
        if key in seen:
            continue
        seen.add(key)
        result.append(dict(obj))
    return result


def _query_hit_counts(query_hits: dict[str, Any]) -> dict[str, int]:
    return {str(query): len(objects or []) for query, objects in query_hits.items()}


def _object_brief(obj: dict[str, Any] | None) -> dict[str, Any] | None:
    if not obj:
        return None
    return {
        "objectId": obj.get("objectId"),
        "objectType": obj.get("objectType"),
        "name": obj.get("name"),
        "visible": obj.get("visible"),
        "distance": obj.get("distance"),
        "pickupable": obj.get("pickupable"),
        "openable": obj.get("openable"),
        "isOpen": obj.get("isOpen"),
        "receptacle": obj.get("receptacle"),
        "toggleable": obj.get("toggleable"),
        "isToggled": obj.get("isToggled"),
        "parentReceptacles": obj.get("parentReceptacles"),
    }


def _step_result_summary(result: StepResult) -> dict[str, Any]:
    return {
        "valid_action": bool(result.valid_action),
        "success": bool(result.success),
        "completed": bool(result.completed),
        "score": result.score,
        "error": result.error,
        "native_action": result.native_action,
        "goal_conditions_met": (result.verification or {}).get("goal_conditions_met"),
        "goal_conditions_total": (result.verification or {}).get("goal_conditions_total"),
    }


def _approach_result_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "valid_action": bool(result.get("valid_action")),
        "error": result.get("error"),
        "events": result.get("events"),
        "final_distance": result.get("final_distance"),
        "stopped_by_budget": bool(result.get("stopped_by_budget")),
    }


def _locate_result_summary(result: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {"found": False, "error": {"kind": "bad_locate_result", "message": str(result)}}
    return {
        "found": bool(result.get("found")),
        "visible_now": bool(result.get("visible_now")),
        "current_object": _object_brief(result.get("current_object")),
        "best_candidate": _object_brief(result.get("best_candidate")),
        "candidate_count": len(result.get("candidates") or []),
        "events": result.get("events"),
        "error": result.get("error"),
    }


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


def _matches(obj: dict[str, Any], query: Any) -> bool:
    q = _norm(query)
    return q in _norm(obj.get("objectId", "")) or q in _norm(obj.get("objectType", "")) or q in _norm(obj.get("name", ""))


def _normalize_queries(queries: Any) -> list[str]:
    if queries is None:
        return []
    if isinstance(queries, str):
        return [queries]
    if isinstance(queries, (list, tuple, set)):
        return [str(query) for query in queries if str(query).strip()]
    return [str(queries)]


def _planar_distance(agent_position: dict[str, Any], target_position: dict[str, Any]) -> float | None:
    try:
        dx = float(target_position["x"]) - float(agent_position["x"])
        dz = float(target_position["z"]) - float(agent_position["z"])
    except (KeyError, TypeError, ValueError):
        return None
    return (dx * dx + dz * dz) ** 0.5


def _coarse_turn_direction(
    agent_position: dict[str, Any],
    target_position: dict[str, Any],
    agent_rotation_y: Any,
) -> str | None:
    import math

    try:
        dx = float(target_position["x"]) - float(agent_position["x"])
        dz = float(target_position["z"]) - float(agent_position["z"])
        current = float(agent_rotation_y) % 360.0
    except (KeyError, TypeError, ValueError):
        return None
    if abs(dx) < 1e-6 and abs(dz) < 1e-6:
        return None
    desired = math.degrees(math.atan2(dx, dz)) % 360.0
    delta = ((desired - current + 540.0) % 360.0) - 180.0
    if abs(delta) <= 45.0:
        return None
    return "right" if delta > 0 else "left"


def _pose_cell(position: dict[str, Any], cell_size: float = 0.25) -> str | None:
    try:
        x = round(float(position["x"]) / cell_size)
        z = round(float(position["z"]) / cell_size)
    except (KeyError, TypeError, ValueError):
        return None
    return f"{x},{z}"


def _round_position(position: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("x", "y", "z"):
        if key in position:
            result[key] = _round(position.get(key))
    return result


def _frontiers_from_cells(cells: dict[str, dict[str, Any]], limit: int = 24) -> list[dict[str, Any]]:
    occupied: set[tuple[int, int]] = set()
    for key in cells:
        try:
            x_text, z_text = key.split(",", 1)
            occupied.add((int(x_text), int(z_text)))
        except ValueError:
            continue
    frontier: list[dict[str, Any]] = []
    for x, z in sorted(occupied):
        for direction, dx, dz in [
            ("north", 0, 1),
            ("east", 1, 0),
            ("south", 0, -1),
            ("west", -1, 0),
        ]:
            neighbor = (x + dx, z + dz)
            if neighbor in occupied:
                continue
            frontier.append({"from_cell": f"{x},{z}", "direction": direction, "candidate_cell": f"{neighbor[0]},{neighbor[1]}"})
            if len(frontier) >= limit:
                return frontier
    return frontier


def _round(value: Any) -> Any:
    return round(float(value), 3) if isinstance(value, (float, int)) else value


def _norm(value: Any) -> str:
    return " ".join(str(value).lower().replace("_", " ").split())


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _safe_frame_label(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value).strip("_") or "frame"
