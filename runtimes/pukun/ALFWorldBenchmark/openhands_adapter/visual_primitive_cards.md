# ALFWorld Visual Primitive Cards

These are the public primitives exposed to the OpenHands workspace for this adapter entry.
Each card follows the RoBench primitive schema: `name`, `signature`, `canonical_family`, `description`, `arguments`, `returns`, `side_effect`, `leakage_level`, `limitations`, and `backend_source`.

## `get_task_context`

- `signature`: `get_task_context()`
- `canonical_family`: `CTX`
- `description`: Return ALFRED task id, split, task type, goal text, public step-by-step language instructions, and public scene name.
- `arguments`: `{}`
- `returns`: `dict`
- `side_effect`: none
- `leakage_level`: L1
- `limitations`: Step-by-step language instructions are public ALFRED annotations, not expert actions. Does not expose expert plans, PDDL params, init action, low_actions, masks, raw traj path, or hidden metadata.
- `backend_source`: official manifest public fields

## `list_actions`

- `signature`: `list_actions()`
- `canonical_family`: `CTX`
- `description`: List the exposed ALFRED visual primitive surface.
- `arguments`: `{}`
- `returns`: `list[str]`
- `side_effect`: none
- `leakage_level`: L1
- `limitations`: This is not the full AI2-THOR action set.
- `backend_source`: harness facade

## `observe`

- `signature`: `observe()`
- `canonical_family`: `PER`
- `description`: Return current frame shape, agent pose, visible object summaries, and inventory.
- `arguments`: `{}`
- `returns`: `dict`
- `side_effect`: none
- `leakage_level`: L1/L2
- `limitations`: Does not return hidden objects, expert plan, or raw image bytes.
- `backend_source`: ThorEnv.last_event metadata

## `get_frame`

- `signature`: `get_frame(label='frame')`
- `canonical_family`: `PER`
- `description`: Save the current RGB frame into the workspace frames/ directory when available and return safe frame metadata.
- `arguments`: `{"label": "Short label used in the saved frame filename."}`
- `returns`: `dict with saved/path/shape/path_boundary`
- `side_effect`: workspace frame copy only
- `leakage_level`: L2 visual evidence
- `limitations`: Does not move the simulator, embed image bytes in trace, or expose backend absolute paths in OpenHands mode.
- `backend_source`: ThorEnv.last_event.frame

## `inspect_current_view`

- `signature`: `inspect_current_view(label=None, remember=True)`
- `canonical_family`: `PER`
- `description`: Capture the current RGB frame, return the current observation and visible objects, and optionally store them in visual memory.
- `arguments`: `{"label": "Optional memory/frame label.", "remember": "Whether to add visible objects to run-local visual memory."}`
- `returns`: `dict with frame, observation, visible_objects, remembered_count, memory_count`
- `side_effect`: workspace frame copy + memory only
- `leakage_level`: L2 visual evidence
- `limitations`: Does not move, search, path-find, or interact with objects.; Only visible objects in the current frame are returned.
- `backend_source`: ThorEnv.last_event.frame + visible metadata

## `detect_objects`

- `signature`: `detect_objects(query=None)`
- `canonical_family`: `DET`
- `description`: Return visible objects, optionally filtered by object type/name/objectId text.
- `arguments`: `{"query": "Optional object query."}`
- `returns`: `list[dict]`
- `side_effect`: none
- `leakage_level`: L2 visible metadata
- `limitations`: Visible objects only.
- `backend_source`: visible metadata

## `remember_visible_objects`

- `signature`: `remember_visible_objects(label=None, query=None)`
- `canonical_family`: `EVD`
- `description`: Store currently visible objects in run-local visual memory, optionally filtered by query and tagged by label.
- `arguments`: `{"label": "Optional memory tag.", "query": "Optional visible-object filter."}`
- `returns`: `dict with remembered_count, memory_count, objects`
- `side_effect`: memory only
- `leakage_level`: L2 visible metadata
- `limitations`: Stores only objects visible at call time; does not reveal hidden scene graph or move the agent.
- `backend_source`: visible metadata + harness memory

## `recall_visible_objects`

- `signature`: `recall_visible_objects(query=None, label=None, limit=50)`
- `canonical_family`: `EVD`
- `description`: Return objects previously stored in visual memory, sorted by latest sighting.
- `arguments`: `{"query": "Optional object type/name/objectId filter.", "label": "Optional memory tag.", "limit": "Maximum returned memories."}`
- `returns`: `list[dict]`
- `side_effect`: none
- `leakage_level`: L2 remembered visible metadata
- `limitations`: Memory can be stale; object may no longer be visible or reachable.
- `backend_source`: harness visual memory

## `read_search_memory`

- `signature`: `read_search_memory(query=None, label=None, limit=50)`
- `canonical_family`: `EVD`
- `description`: Alias-style memory reader for search traces: return remembered visible objects filtered by query/label.
- `arguments`: `{"query": "Optional object type/name/objectId filter.", "label": "Optional memory tag.", "limit": "Maximum returned memories."}`
- `returns`: `list[dict]`
- `side_effect`: none
- `leakage_level`: L2 remembered visible metadata
- `limitations`: Reads only objects previously seen through perception/search primitives.; Memory can be stale.
- `backend_source`: harness visual memory

## `read_observed_spatial_map`

- `signature`: `read_observed_spatial_map(query=None, limit=50)`
- `canonical_family`: `STATE`
- `description`: Return an observed-only spatial memory summary: current pose, visited coarse cells, blocked moves, frontier hints, and seen object memory.
- `arguments`: `{"query": "Optional filter for seen objects.", "limit": "Maximum blocked moves / object memories returned."}`
- `returns`: `dict with current_agent, visited_cells, blocked_moves, frontiers, seen_objects, boundary`
- `side_effect`: memory read only
- `leakage_level`: L2 observed metadata + L3 navigation memory
- `limitations`: Only summarizes poses and objects observed through prior primitive calls.; Does not expose hidden objects, full reachable map, target pose, shortest path, benchmark walkthrough data, masks, or low_actions.; Frontiers are coarse heuristic neighbors of visited cells, not simulator-verified reachable positions.
- `backend_source`: harness observed pose/object memory

## `scan_scene`

- `signature`: `scan_scene(queries=None, rotations=4, include_tilts=True, remember=True)`
- `canonical_family`: `PER`
- `description`: Perform a bounded in-place rotate/look scan, optionally remember seen objects, and return query hits.
- `arguments`: `{"queries": "String or list of object queries.", "rotations": "Number of left rotations, capped by harness.", "include_tilts": "Whether to look up/down during scan.", "remember": "Whether to store seen objects in visual memory."}`
- `returns`: `dict with observations, query_hits, all_seen, memory_count`
- `side_effect`: bounded rotate/look native actions
- `leakage_level`: L2 visible metadata + L3 navigation
- `limitations`: No path finding, no object interaction, no expert replay, and only objects visible during the scan are returned.
- `backend_source`: observe + RotateLeft + LookUp/LookDown

## `search_scene`

- `signature`: `search_scene(queries=None, rounds=3, scan_rotations=4, include_tilts=True, remember=True)`
- `canonical_family`: `PER`
- `description`: Run several bounded scan_scene rounds, moving ahead slightly between rounds, and return aggregate query hits.
- `arguments`: `{"queries": "String or list of object queries.", "rounds": "Number of scan/move rounds, capped by harness.", "scan_rotations": "Rotate count per scan.", "include_tilts": "Whether each scan looks up/down.", "remember": "Whether to store seen objects in visual memory."}`
- `returns`: `dict with scans, movement_events, query_hits, all_seen, memory_count`
- `side_effect`: bounded rotate/look/move native actions
- `leakage_level`: L2 visible metadata + L3 navigation
- `limitations`: No object interaction, no path planning, no target manipulation, no expert replay; it may miss objects and may move into unhelpful poses.
- `backend_source`: scan_scene + MoveAhead + RotateRight-if-blocked

## `explore_room`

- `signature`: `explore_room(queries=None, step_budget=32, include_tilts=True, remember=True)`
- `canonical_family`: `PER`
- `description`: Run a bounded generic room exploration pattern while recording visible-object memory and query hits.
- `arguments`: `{"queries": "String or list of object queries.", "step_budget": "Maximum native move/rotate/look actions, capped by harness.", "include_tilts": "Whether to periodically look up/down.", "remember": "Whether to store seen objects in visual memory."}`
- `returns`: `dict with observations, movement_events, query_hits, all_seen, memory_count`
- `side_effect`: bounded move/rotate/look native actions
- `leakage_level`: L2 visible metadata + L3 navigation
- `limitations`: No object interaction, no hidden map, no shortest-path planner, no target-specific route, and no expert replay; exploration can still miss objects.
- `backend_source`: visible metadata + MoveAhead/RotateLeft/LookUp/LookDown

## `ground_object`

- `signature`: `ground_object(query)`
- `canonical_family`: `DET`
- `description`: Resolve a unique visible object by objectId, type, or name.
- `arguments`: `{"query": "Unique visible object query."}`
- `returns`: `dict or error`
- `side_effect`: none
- `leakage_level`: L2 visible metadata
- `limitations`: Ambiguous names fail and return visible candidates.
- `backend_source`: visible metadata

## `query_object_state`

- `signature`: `query_object_state(obj)`
- `canonical_family`: `STATE`
- `description`: Return safe state fields for one visible or inventory object.
- `arguments`: `{"obj": "objectId, object dict, type, or name."}`
- `returns`: `dict or error`
- `side_effect`: none
- `leakage_level`: L2 metadata
- `limitations`: Does not expose full hidden scene graph.
- `backend_source`: visible/inventory metadata

## `query_inventory`

- `signature`: `query_inventory()`
- `canonical_family`: `STATE`
- `description`: Return objects currently held by the agent.
- `arguments`: `{}`
- `returns`: `list[dict]`
- `side_effect`: none
- `leakage_level`: L1
- `limitations`: Only current inventory.
- `backend_source`: metadata inventoryObjects

## `move_ahead`

- `signature`: `move_ahead()`
- `canonical_family`: `NAV`
- `description`: Move the agent one ALFRED/THOR step forward.
- `arguments`: `{}`
- `returns`: `StepResult`
- `side_effect`: one native action
- `leakage_level`: L3
- `limitations`: No path finding.
- `backend_source`: MoveAhead

## `rotate`

- `signature`: `rotate(direction)`
- `canonical_family`: `NAV`
- `description`: Rotate the agent in place.
- `arguments`: `{"direction": "left or right."}`
- `returns`: `StepResult`
- `side_effect`: one native action
- `leakage_level`: L3
- `limitations`: No automatic target alignment.
- `backend_source`: RotateLeft/RotateRight

## `look`

- `signature`: `look(direction)`
- `canonical_family`: `NAV`
- `description`: Tilt the camera.
- `arguments`: `{"direction": "up or down."}`
- `returns`: `StepResult`
- `side_effect`: one native action
- `leakage_level`: L3
- `limitations`: No object interaction.
- `backend_source`: LookUp/LookDown

## `approach_object`

- `signature`: `approach_object(obj, max_steps=3, stop_distance=1.25)`
- `canonical_family`: `NAV`
- `description`: Coarsely turn and move toward a visible or remembered object position.
- `arguments`: `{"obj": "Visible or remembered object dict/objectId/type/name.", "max_steps": "Small bounded action budget.", "stop_distance": "Distance threshold."}`
- `returns`: `dict with events, final_distance, final_observation`
- `side_effect`: bounded rotate/move native actions
- `leakage_level`: L2 remembered visible metadata + L3 navigation
- `limitations`: No path finding, no obstacle map, no hidden object search, no object interaction; only works for visible or previously remembered objects.
- `backend_source`: visible metadata/memory + RotateLeft/RotateRight + MoveAhead

## `open_object`

- `signature`: `open_object(obj)`
- `canonical_family`: `HACT`
- `description`: Open one visible openable object.
- `arguments`: `{"obj": "Visible object reference."}`
- `returns`: `StepResult`
- `side_effect`: one native action
- `leakage_level`: L3
- `limitations`: Fails if object is not visible/openable.
- `backend_source`: OpenObject

## `close_object`

- `signature`: `close_object(obj)`
- `canonical_family`: `HACT`
- `description`: Close one visible openable object.
- `arguments`: `{"obj": "Visible object reference."}`
- `returns`: `StepResult`
- `side_effect`: one native action
- `leakage_level`: L3
- `limitations`: Fails if object is not visible/openable.
- `backend_source`: CloseObject

## `pickup_object`

- `signature`: `pickup_object(obj)`
- `canonical_family`: `HACT`
- `description`: Pick up one visible pickupable object.
- `arguments`: `{"obj": "Visible object reference."}`
- `returns`: `StepResult`
- `side_effect`: one native action
- `leakage_level`: L3
- `limitations`: Fails if object is not visible/pickupable.
- `backend_source`: PickupObject

## `put_object`

- `signature`: `put_object(obj, receptacle)`
- `canonical_family`: `HACT`
- `description`: Put the held object into/on a visible receptacle.
- `arguments`: `{"obj": "Held object reference.", "receptacle": "Visible receptacle reference."}`
- `returns`: `StepResult`
- `side_effect`: one native action
- `leakage_level`: L3
- `limitations`: No automatic pickup; object must already be held.
- `backend_source`: PutObject

## `toggle_object`

- `signature`: `toggle_object(obj, on=True)`
- `canonical_family`: `HACT`
- `description`: Toggle a visible toggleable object on or off.
- `arguments`: `{"obj": "Visible object reference.", "on": "True for on, False for off."}`
- `returns`: `StepResult`
- `side_effect`: one native action
- `leakage_level`: L3
- `limitations`: Fails if object is not visible/toggleable.
- `backend_source`: ToggleObjectOn/ToggleObjectOff

## `slice_object`

- `signature`: `slice_object(obj)`
- `canonical_family`: `HACT`
- `description`: Slice a visible sliceable object when holding a knife.
- `arguments`: `{"obj": "Visible sliceable object reference."}`
- `returns`: `StepResult`
- `side_effect`: one native action
- `leakage_level`: L3
- `limitations`: Does not auto-pickup knife or navigate.
- `backend_source`: SliceObject

## `write_evidence`

- `signature`: `write_evidence(key, value)`
- `canonical_family`: `EVD`
- `description`: Store run-local notes for trace analysis.
- `arguments`: `{"key": "Evidence key.", "value": "JSON-serializable value."}`
- `returns`: `dict`
- `side_effect`: none
- `leakage_level`: L1
- `limitations`: Does not affect simulator state.
- `backend_source`: harness memory

## `read_evidence`

- `signature`: `read_evidence()`
- `canonical_family`: `EVD`
- `description`: Read current run-local notes.
- `arguments`: `{}`
- `returns`: `dict`
- `side_effect`: none
- `leakage_level`: L1
- `limitations`: Current run only.
- `backend_source`: harness memory

## `check_progress_public`

- `signature`: `check_progress_public()`
- `canonical_family`: `VERIFY`
- `description`: Return public progress, inventory, visible-object count, memory count, and step usage.
- `arguments`: `{}`
- `returns`: `dict`
- `side_effect`: none
- `leakage_level`: L1 verifier summary
- `limitations`: Does not reveal expert low_actions, masks, hidden target ids, or private trajectory fields.
- `backend_source`: ThorEnv goal-condition checker + public observation summary

## `check_success`

- `signature`: `check_success()`
- `canonical_family`: `VERIFY`
- `description`: Return official ALFRED goal condition progress and success.
- `arguments`: `{}`
- `returns`: `dict`
- `side_effect`: none
- `leakage_level`: L1 verifier summary
- `limitations`: Does not reveal expert low_actions, masks, or hidden target object ids.
- `backend_source`: ThorEnv.get_goal_conditions_met/get_goal_satisfied
