# ALFWorld Text Primitive Cards

These are the public primitives exposed to the OpenHands workspace for this adapter entry.
Each card follows the RoBench primitive schema: `name`, `signature`, `canonical_family`, `description`, `arguments`, `returns`, `side_effect`, `leakage_level`, `limitations`, and `backend_source`.

## `get_task_context`

- `signature`: `get_task_context()`
- `canonical_family`: `CTX`
- `description`: Return benchmark metadata, task id, task type, official goal text, and step budget.
- `arguments`: `{}`
- `returns`: `dict`
- `side_effect`: none
- `leakage_level`: L1 public task metadata
- `limitations`: Does not expose expert plans, PDDL facts, hidden state, or walkthroughs.
- `backend_source`: TaskRecord + official game grammar goal text

## `observe_text_state`

- `signature`: `observe_text_state()`
- `canonical_family`: `CTX`
- `description`: Return the latest text observation from the ALFWorld TextWorld environment.
- `arguments`: `{}`
- `returns`: `str`
- `side_effect`: none
- `leakage_level`: L1 current observation
- `limitations`: Only returns what the text environment currently reveals.
- `backend_source`: AlfredTWEnv reset/step observation cache

## `list_actions`

- `signature`: `list_actions()`
- `canonical_family`: `CTX`
- `description`: Return currently admissible ALFWorld native text commands.
- `arguments`: `{}`
- `returns`: `list[str]`
- `side_effect`: none
- `leakage_level`: L1 environment affordance
- `limitations`: Use it to disambiguate object indices such as shelf 1 or desk 2.
- `backend_source`: AlfredTWEnv infos['admissible_commands']

## `match_actions`

- `signature`: `match_actions(intent=None, object_name=None, receptacle_name=None, include=None, limit=20)`
- `canonical_family`: `CTX`
- `description`: Return currently admissible native commands that match a high-level intent and optional object/receptacle text.
- `arguments`: `{"intent": "Optional intent such as go_to, pickup, place, open, close, clean, heat, cool, examine.", "object_name": "Optional object text that must appear in the native command.", "receptacle_name": "Optional target receptacle/location text that must appear in the native command.", "include": "Optional extra substring or list of substrings.", "limit": "Maximum returned matches."}`
- `returns`: `dict with query, total_matches, and scored action candidates`
- `side_effect`: none
- `leakage_level`: L1 environment affordance
- `limitations`: Read-only matcher; it does not execute, navigate, open containers, or choose a plan.; Returned actions are current-state only and must still be copied exactly into step_text_action(action).
- `backend_source`: current admissible command list

## `examine_object`

- `signature`: `examine_object(name)`
- `canonical_family`: `PER`
- `description`: Examine one currently admissible visible object, receptacle, or surface matching name.
- `arguments`: `{"name": "Object, receptacle, or surface name, for example 'desk 1' or 'sidetable 2'."}`
- `returns`: `StepResult; use result.valid_action to check action execution, while result.success means whole-task success.`
- `side_effect`: one native text action
- `leakage_level`: L3 action
- `limitations`: Thin wrapper only: executes at most one native examine command and does not search automatically.; Returns an error if no unique admissible examine command matches.
- `backend_source`: current admissible examine command matched to AlfredTWEnv.step

## `look`

- `signature`: `look()`
- `canonical_family`: `NAV`
- `description`: Execute the native look action.
- `arguments`: `{}`
- `returns`: `StepResult; use result.valid_action to check action execution, while result.success means whole-task success.`
- `side_effect`: one native text action
- `leakage_level`: L3 action
- `limitations`: Consumes one environment step.
- `backend_source`: AlfredTWEnv.step('look')

## `inventory`

- `signature`: `inventory()`
- `canonical_family`: `STATE`
- `description`: Execute the native inventory action.
- `arguments`: `{}`
- `returns`: `StepResult; use result.valid_action to check action execution, while result.success means whole-task success.`
- `side_effect`: one native text action
- `leakage_level`: L3 action
- `limitations`: Consumes one environment step.
- `backend_source`: AlfredTWEnv.step('inventory')

## `go_to`

- `signature`: `go_to(name)`
- `canonical_family`: `NAV`
- `description`: Move to one currently admissible location whose text command contains name.
- `arguments`: `{"name": "Location name or partial name, for example 'desk 1' or 'shelf 2'."}`
- `returns`: `StepResult; use result.valid_action to check action execution, while result.success means whole-task success.`
- `side_effect`: one native text action
- `leakage_level`: L3 action
- `limitations`: Thin wrapper only: executes at most one native command and does not search automatically.
- `backend_source`: current admissible command matched to AlfredTWEnv.step

## `open_object`

- `signature`: `open_object(name)`
- `canonical_family`: `HACT`
- `description`: Open one currently admissible object or receptacle matching name.
- `arguments`: `{"name": "Object or receptacle name."}`
- `returns`: `StepResult; use result.valid_action to check action execution, while result.success means whole-task success.`
- `side_effect`: one native text action
- `leakage_level`: L3 action
- `limitations`: Returns an error if no unique admissible open command matches.
- `backend_source`: current admissible open command matched to AlfredTWEnv.step

## `close_object`

- `signature`: `close_object(name)`
- `canonical_family`: `HACT`
- `description`: Close one currently admissible object or receptacle matching name.
- `arguments`: `{"name": "Object or receptacle name."}`
- `returns`: `StepResult; use result.valid_action to check action execution, while result.success means whole-task success.`
- `side_effect`: one native text action
- `leakage_level`: L3 action
- `limitations`: Returns an error if no unique admissible close command matches.
- `backend_source`: current admissible close command matched to AlfredTWEnv.step

## `pickup_object`

- `signature`: `pickup_object(name)`
- `canonical_family`: `HACT`
- `description`: Pick up one currently admissible object matching name.
- `arguments`: `{"name": "Object name, for example 'pencil' or 'apple 1'."}`
- `returns`: `StepResult; use result.valid_action to check action execution, while result.success means whole-task success.`
- `side_effect`: one native text action
- `leakage_level`: L3 action
- `limitations`: Requires the object to be available in the current state.
- `backend_source`: current admissible pickup command matched to AlfredTWEnv.step

## `place_object`

- `signature`: `place_object(obj, receptacle)`
- `canonical_family`: `HACT`
- `description`: Place or move a held object into/on one currently admissible receptacle matching receptacle.
- `arguments`: `{"obj": "Held object name, for example 'pencil'.", "receptacle": "Target receptacle/location name, for example 'shelf 1'."}`
- `returns`: `StepResult; use result.valid_action to check action execution, while result.success means whole-task success.`
- `side_effect`: one native text action
- `leakage_level`: L3 action
- `limitations`: Thin wrapper only: does not navigate before placing and executes at most one native command.
- `backend_source`: current admissible put/place command matched to AlfredTWEnv.step

## `toggle_object`

- `signature`: `toggle_object(name)`
- `canonical_family`: `HACT`
- `description`: Use, toggle, turn on, or turn off one currently admissible object matching name.
- `arguments`: `{"name": "Object name."}`
- `returns`: `StepResult; use result.valid_action to check action execution, while result.success means whole-task success.`
- `side_effect`: one native text action
- `leakage_level`: L3 action
- `limitations`: Returns an error if no unique admissible toggle/use command matches.
- `backend_source`: current admissible use/toggle command matched to AlfredTWEnv.step

## `clean_object`

- `signature`: `clean_object(obj)`
- `canonical_family`: `HACT`
- `description`: Clean or wash one held object when the current state permits it.
- `arguments`: `{"obj": "Object name."}`
- `returns`: `StepResult; use result.valid_action to check action execution, while result.success means whole-task success.`
- `side_effect`: one native text action
- `leakage_level`: L3 action
- `limitations`: Does not navigate to a sink and does not pick up the object automatically.
- `backend_source`: current admissible clean/wash command matched to AlfredTWEnv.step

## `heat_object`

- `signature`: `heat_object(obj)`
- `canonical_family`: `HACT`
- `description`: Heat one held object when the current state permits it.
- `arguments`: `{"obj": "Object name."}`
- `returns`: `StepResult; use result.valid_action to check action execution, while result.success means whole-task success.`
- `side_effect`: one native text action
- `leakage_level`: L3 action
- `limitations`: Does not navigate to a microwave and does not pick up the object automatically.
- `backend_source`: current admissible heat command matched to AlfredTWEnv.step

## `cool_object`

- `signature`: `cool_object(obj)`
- `canonical_family`: `HACT`
- `description`: Cool one held object when the current state permits it.
- `arguments`: `{"obj": "Object name."}`
- `returns`: `StepResult; use result.valid_action to check action execution, while result.success means whole-task success.`
- `side_effect`: one native text action
- `leakage_level`: L3 action
- `limitations`: Does not navigate to a fridge and does not pick up the object automatically.
- `backend_source`: current admissible cool command matched to AlfredTWEnv.step

## `check_success`

- `signature`: `check_success()`
- `canonical_family`: `VERIFY`
- `description`: Return the ALFWorld native verifier result.
- `arguments`: `{}`
- `returns`: `dict with success, score, done, and source`
- `side_effect`: none
- `leakage_level`: L1 verifier summary
- `limitations`: Verifier reads only native ALFWorld success signals.
- `backend_source`: AlfredTWEnv infos['won'], score, done

## `write_evidence`

- `signature`: `write_evidence(key, value)`
- `canonical_family`: `EVD`
- `description`: Store intermediate evidence in the trace-local memory.
- `arguments`: `{"key": "Evidence key.", "value": "JSON-serializable evidence value."}`
- `returns`: `dict`
- `side_effect`: trace-local memory write
- `leakage_level`: L1
- `limitations`: Evidence helps trace reasoning but does not directly change ALFWorld state.
- `backend_source`: RoBench trace memory

## `read_evidence`

- `signature`: `read_evidence()`
- `canonical_family`: `EVD`
- `description`: Read evidence previously written by this task run.
- `arguments`: `{}`
- `returns`: `dict`
- `side_effect`: none
- `leakage_level`: L1
- `limitations`: Only includes evidence written through write_evidence in the current run.
- `backend_source`: RoBench trace memory
