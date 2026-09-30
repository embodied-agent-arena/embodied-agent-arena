# DiscoveryWorld Text/JSON Primitive Cards

These are the public primitives exposed to the OpenHands workspace for this adapter entry.
Each card follows the RoBench primitive schema: `name`, `signature`, `canonical_family`, `description`, `arguments`, `returns`, `side_effect`, `leakage_level`, `limitations`, and `backend_source`.

## `get_task_context`

- `signature`: `get_task_context()`
- `canonical_family`: `CTX`
- `description`: Return benchmark metadata, scenario, difficulty, seed, official goal text, and step budget.
- `arguments`: `{}`
- `returns`: `dict`
- `side_effect`: none
- `leakage_level`: L1
- `limitations`: Does not expose scorecard internals, world history, or hidden state.
- `backend_source`: manifest + current taskProgress description

## `observe_world`

- `signature`: `observe_world()`
- `canonical_family`: `CTX`
- `description`: Return the latest safe JSON observation summary.
- `arguments`: `{}`
- `returns`: `dict`
- `side_effect`: none
- `leakage_level`: L1
- `limitations`: Vision base64 and full world object lists are not returned.
- `backend_source`: DiscoveryWorldAPI.getAgentObservation

## `list_known_actions`

- `signature`: `list_known_actions()`
- `canonical_family`: `CTX`
- `description`: Return native action descriptions supported by the v1 primitive facade.
- `arguments`: `{}`
- `returns`: `dict`
- `side_effect`: none
- `leakage_level`: L1
- `limitations`: Does not include TELEPORT_TO_OBJECT or first-round Discovery Feed actions.
- `backend_source`: DiscoveryWorldAPI.listKnownActions

## `get_action_schema`

- `signature`: `get_action_schema()`
- `canonical_family`: `CTX`
- `description`: Return the primitive-level argument schema and native payload shape for exposed DiscoveryWorld actions.
- `arguments`: `{}`
- `returns`: `dict`
- `side_effect`: none
- `leakage_level`: L1
- `limitations`: Schema only; it does not execute or reveal a plan.
- `backend_source`: RoBench primitive facade schema

## `list_accessible_objects`

- `signature`: `list_accessible_objects()`
- `canonical_family`: `STATE`
- `description`: List currently accessible objects that can be interacted with.
- `arguments`: `{}`
- `returns`: `list[dict]`
- `side_effect`: none
- `leakage_level`: L1
- `limitations`: Object names may be ambiguous; use uuid when multiple objects share a name.
- `backend_source`: observation.ui.accessibleEnvironmentObjects

## `list_nearby_objects`

- `signature`: `list_nearby_objects(query=None, max_distance=None)`
- `canonical_family`: `STATE`
- `description`: List nearby objects grouped into a flat direction-aware view for navigation and search.
- `arguments`: `{"query": "Optional name/description/uuid substring.", "max_distance": "Optional maximum nearby distance."}`
- `returns`: `list[dict]`
- `side_effect`: none
- `leakage_level`: L1/L2
- `limitations`: Nearby does not mean accessible; call list_accessible_objects() before manipulation.
- `backend_source`: observation.ui.nearbyObjects

## `list_inventory`

- `signature`: `list_inventory()`
- `canonical_family`: `STATE`
- `description`: List objects currently in inventory.
- `arguments`: `{}`
- `returns`: `list[dict]`
- `side_effect`: none
- `leakage_level`: L1
- `limitations`: Only returns inventory visible to the agent.
- `backend_source`: observation.ui.inventoryObjects

## `validate_action_call`

- `signature`: `validate_action_call(primitive_name, arguments=None, **kwargs)`
- `canonical_family`: `STATE`
- `description`: Dry-run wrapper argument validation and object resolution before taking a side-effecting action.
- `arguments`: `{"primitive_name": "Primitive name such as move_direction, pickup_object, or put_object.", "arguments": "Optional dict of arguments.", "**kwargs": "Alternative keyword arguments, for example direction='north'."}`
- `returns`: `dict with ok/error/native_action_json/resolved candidates`
- `side_effect`: none
- `leakage_level`: L1/L2
- `limitations`: Does not call the simulator or tick time.; A valid wrapper payload can still fail in the native simulator because of dynamic preconditions.
- `backend_source`: RoBench primitive facade resolver

## `list_teleport_locations`

- `signature`: `list_teleport_locations()`
- `canonical_family`: `NAV`
- `description`: List official named teleport locations for efficient smoke/navigation.
- `arguments`: `{}`
- `returns`: `dict`
- `side_effect`: none
- `leakage_level`: L3
- `limitations`: Teleport is an allowed v1 navigation helper and must be trace-marked when used.
- `backend_source`: DiscoveryWorldAPI.listTeleportLocationsDict

## `move_direction`

- `signature`: `move_direction(direction)`
- `canonical_family`: `NAV`
- `description`: Move one tile north, east, south, or west.
- `arguments`: `{"direction": "One of north, east, south, west."}`
- `returns`: `StepResult`
- `side_effect`: one native action + tick
- `leakage_level`: L3
- `limitations`: Does not path-find or retry.; direction is a string, not an object uuid; use validate_action_call before execution if unsure.
- `backend_source`: MOVE_DIRECTION

## `rotate_direction`

- `signature`: `rotate_direction(direction)`
- `canonical_family`: `NAV`
- `description`: Rotate to face north, east, south, or west.
- `arguments`: `{"direction": "One of north, east, south, west."}`
- `returns`: `StepResult`
- `side_effect`: one native action + tick
- `leakage_level`: L3
- `limitations`: Does not move.
- `backend_source`: ROTATE_DIRECTION

## `teleport_to_location`

- `signature`: `teleport_to_location(location_name)`
- `canonical_family`: `NAV`
- `description`: Teleport to one official named location.
- `arguments`: `{"location_name": "Name from list_teleport_locations()."}`
- `returns`: `StepResult`
- `side_effect`: one native action + tick
- `leakage_level`: L3
- `limitations`: No object teleport. Invalid names fail without guessing.
- `backend_source`: TELEPORT_TO_LOCATION

## `pickup_object`

- `signature`: `pickup_object(obj)`
- `canonical_family`: `HACT`
- `description`: Pick up one accessible object.
- `arguments`: `{"obj": "uuid or unique visible name."}`
- `returns`: `StepResult`
- `side_effect`: one native action + tick
- `leakage_level`: L3
- `limitations`: No search or navigation.
- `backend_source`: PICKUP

## `drop_object`

- `signature`: `drop_object(obj)`
- `canonical_family`: `HACT`
- `description`: Drop one inventory object.
- `arguments`: `{"obj": "uuid or unique inventory name."}`
- `returns`: `StepResult`
- `side_effect`: one native action + tick
- `leakage_level`: L3
- `limitations`: Does not choose placement.
- `backend_source`: DROP

## `put_object`

- `signature`: `put_object(obj, target)`
- `canonical_family`: `HACT`
- `description`: Put an inventory object in/on another accessible object or give it to an agent.
- `arguments`: `{"obj": "uuid or unique inventory name.", "target": "uuid or unique accessible target."}`
- `returns`: `StepResult`
- `side_effect`: one native action + tick
- `leakage_level`: L3
- `limitations`: No navigation or search.
- `backend_source`: PUT

## `open_object`

- `signature`: `open_object(obj)`
- `canonical_family`: `HACT`
- `description`: Open one accessible object.
- `arguments`: `{"obj": "uuid or unique visible name."}`
- `returns`: `StepResult`
- `side_effect`: one native action + tick
- `leakage_level`: L3
- `limitations`: Fails if object is not accessible/openable.
- `backend_source`: OPEN

## `close_object`

- `signature`: `close_object(obj)`
- `canonical_family`: `HACT`
- `description`: Close one accessible object.
- `arguments`: `{"obj": "uuid or unique visible name."}`
- `returns`: `StepResult`
- `side_effect`: one native action + tick
- `leakage_level`: L3
- `limitations`: Fails if object is not accessible/closeable.
- `backend_source`: CLOSE

## `activate_object`

- `signature`: `activate_object(obj)`
- `canonical_family`: `HACT`
- `description`: Activate one accessible object.
- `arguments`: `{"obj": "uuid or unique visible name."}`
- `returns`: `StepResult`
- `side_effect`: one native action + tick
- `leakage_level`: L3
- `limitations`: No automatic tool selection.
- `backend_source`: ACTIVATE

## `deactivate_object`

- `signature`: `deactivate_object(obj)`
- `canonical_family`: `HACT`
- `description`: Deactivate one accessible object.
- `arguments`: `{"obj": "uuid or unique visible name."}`
- `returns`: `StepResult`
- `side_effect`: one native action + tick
- `leakage_level`: L3
- `limitations`: No automatic tool selection.
- `backend_source`: DEACTIVATE

## `use_object`

- `signature`: `use_object(obj, target)`
- `canonical_family`: `HACT`
- `description`: Use one object on another object.
- `arguments`: `{"obj": "uuid or unique visible/inventory name.", "target": "uuid or unique visible/inventory name."}`
- `returns`: `StepResult`
- `side_effect`: one native action + tick
- `leakage_level`: L3
- `limitations`: Both args are required by native USE.
- `backend_source`: USE

## `read_object`

- `signature`: `read_object(obj)`
- `canonical_family`: `STATE`
- `description`: Read one accessible or inventory object.
- `arguments`: `{"obj": "uuid or unique name."}`
- `returns`: `StepResult`
- `side_effect`: one native action + tick
- `leakage_level`: L3
- `limitations`: Only reads objects the environment allows reading.
- `backend_source`: READ

## `eat_object`

- `signature`: `eat_object(obj)`
- `canonical_family`: `HACT`
- `description`: Eat one inventory or accessible object.
- `arguments`: `{"obj": "uuid or unique name."}`
- `returns`: `StepResult`
- `side_effect`: one native action + tick
- `leakage_level`: L3
- `limitations`: No health/goal interpretation.
- `backend_source`: EAT

## `wait`

- `signature`: `wait()`
- `canonical_family`: `STATE`
- `description`: Advance the world by one tick without a native action.
- `arguments`: `{}`
- `returns`: `StepResult`
- `side_effect`: tick only
- `leakage_level`: L3
- `limitations`: Useful for moving agents; still consumes budget.
- `backend_source`: tick

## `talk_to`

- `signature`: `talk_to(agent)`
- `canonical_family`: `DIALOG`
- `description`: Talk to one accessible agent.
- `arguments`: `{"agent": "uuid or unique visible agent name."}`
- `returns`: `StepResult`
- `side_effect`: one native action + tick
- `leakage_level`: L3
- `limitations`: Dialog choices require choose_dialog_option().
- `backend_source`: TALK

## `choose_dialog_option`

- `signature`: `choose_dialog_option(option_index)`
- `canonical_family`: `DIALOG`
- `description`: Choose a numbered dialog option while in dialog mode.
- `arguments`: `{"option_index": "1-based integer option index from observation['dialog_box']['dialogOptions']."}`
- `returns`: `StepResult with valid_action, error, success, completed, score, verification`
- `side_effect`: one dialog action + tick
- `leakage_level`: L3
- `limitations`: Only valid while observation.dialog_box.is_in_dialog is true.
- `backend_source`: chosen_dialog_option_int

## `write_evidence`

- `signature`: `write_evidence(key, value)`
- `canonical_family`: `EVD`
- `description`: Store trace-local evidence.
- `arguments`: `{"key": "Evidence key.", "value": "JSON-serializable value."}`
- `returns`: `dict`
- `side_effect`: none
- `leakage_level`: L1
- `limitations`: Does not alter the DiscoveryWorld environment.; Evidence keys that look like private/oracle/backend fields are renamed before trace storage.
- `backend_source`: harness memory

## `read_evidence`

- `signature`: `read_evidence()`
- `canonical_family`: `EVD`
- `description`: Read evidence written in this task run.
- `arguments`: `{}`
- `returns`: `dict`
- `side_effect`: none
- `leakage_level`: L1
- `limitations`: Only current run evidence.
- `backend_source`: harness memory

## `check_success`

- `signature`: `check_success()`
- `canonical_family`: `VERIFY`
- `description`: Return safe native verifier summary.
- `arguments`: `{}`
- `returns`: `dict`
- `side_effect`: none
- `leakage_level`: L1 verifier summary
- `limitations`: Does not expose scoreCard, criticalHypotheses, or associated notes.
- `backend_source`: getTaskScorecard + areTasksComplete
