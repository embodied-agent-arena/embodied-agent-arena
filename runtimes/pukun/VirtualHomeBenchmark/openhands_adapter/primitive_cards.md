# VirtualHome Symbolic Primitive Cards

These are the public primitives exposed to the OpenHands workspace for this adapter entry.
Each card follows the RoBench primitive schema: `name`, `signature`, `canonical_family`, `description`, `arguments`, `returns`, `side_effect`, `leakage_level`, `limitations`, and `backend_source`.

## `get_task_context`

- `signature`: `get_task_context()`
- `canonical_family`: `CTX`
- `description`: Return task id, goal text, suite, action budget, and success predicate kind.
- `arguments`: `{}`
- `returns`: `dict`
- `side_effect`: none
- `leakage_level`: L1
- `limitations`: Does not expose the expected plan.
- `backend_source`: local manifest public fields

## `list_actions`

- `signature`: `list_actions(query=None, limit=80)`
- `canonical_family`: `CTX`
- `description`: List candidate one-step VirtualHome program actions, optionally filtered by text.
- `arguments`: `{"query": "Optional substring such as 'keyboard', 'chair', or 'home_office'.", "limit": "Maximum returned actions, capped at 200."}`
- `returns`: `list[str]`
- `side_effect`: none
- `leakage_level`: L1
- `limitations`: This is a bounded candidate pool, not an ordered gold plan.; Use query/limit instead of printing huge action lists.
- `backend_source`: suite action candidate pool

## `list_executable_actions`

- `signature`: `list_executable_actions(query=None, limit=40)`
- `canonical_family`: `STATE`
- `description`: List candidate program actions that are currently executable under the Evolving Graph preconditions.
- `arguments`: `{"query": "Optional substring such as an object class/id.", "limit": "Maximum executable actions returned, capped at 100."}`
- `returns`: `list[str]`
- `side_effect`: none
- `leakage_level`: L2/L3 precondition probe
- `limitations`: This is not a gold plan and is not ordered by task relevance.; It dry-runs candidates on a state copy and does not change the benchmark state.
- `backend_source`: VirtualHome ScriptExecutor.execute_one_step on copied state

## `query_symbolic_state`

- `signature`: `query_symbolic_state(scope=None)`
- `canonical_family`: `STATE`
- `description`: Return compact symbolic nodes and relations from the current Evolving Graph state.
- `arguments`: `{"scope": "Optional object class/id substring."}`
- `returns`: `dict`
- `side_effect`: none
- `leakage_level`: L2 symbolic state
- `limitations`: No Unity pixels or hidden gold program are returned.
- `backend_source`: VirtualHome Evolving Graph state

## `validate_program_step`

- `signature`: `validate_program_step(action_line)`
- `canonical_family`: `STATE`
- `description`: Dry-run one VirtualHome program line against the current symbolic state and return validity/error without state mutation.
- `arguments`: `{"action_line": "One candidate line, usually copied from list_actions() or list_executable_actions()."}`
- `returns`: `dict with valid_action/error/verifier summary and compact after-state when valid`
- `side_effect`: none
- `leakage_level`: L2/L3 precondition probe
- `limitations`: Does not execute the action in the real benchmark state.; A valid single step is not necessarily task-solving.
- `backend_source`: VirtualHome ScriptExecutor.execute_one_step on copied state

## `explain_action_preconditions`

- `signature`: `explain_action_preconditions(action_line)`
- `canonical_family`: `STATE`
- `description`: Return validation plus the relevant current graph nodes/relations for the objects mentioned in an action line.
- `arguments`: `{"action_line": "One candidate VirtualHome action line."}`
- `returns`: `dict`
- `side_effect`: none
- `leakage_level`: L2 symbolic state
- `limitations`: Uses executor error text and current graph context; it does not reveal hidden expert programs.
- `backend_source`: VirtualHome Evolving Graph state + executor error

## `execute_program_step`

- `signature`: `execute_program_step(action_line)`
- `canonical_family`: `HACT`
- `description`: Execute exactly one VirtualHome program line, such as '[Open] <fridge> (3)'.
- `arguments`: `{"action_line": "One supported VirtualHome action line."}`
- `returns`: `dict with valid_action/valid, error, verifier summary, and compact after-state around the character/action object`
- `side_effect`: one symbolic transition
- `leakage_level`: L3
- `limitations`: Does not execute multi-step plans automatically.; Must be called from solve.py, never from python -c or temporary scripts.
- `backend_source`: VirtualHome ScriptExecutor.execute_one_step

## `write_evidence`

- `signature`: `write_evidence(key, value)`
- `canonical_family`: `EVD`
- `description`: Store trace-local evidence for later inspection.
- `arguments`: `{"key": "Evidence key.", "value": "JSON-serializable value."}`
- `returns`: `dict`
- `side_effect`: none
- `leakage_level`: L1
- `limitations`: Does not change symbolic state.
- `backend_source`: harness memory

## `read_evidence`

- `signature`: `read_evidence()`
- `canonical_family`: `EVD`
- `description`: Read trace-local evidence accumulated by the generated code.
- `arguments`: `{}`
- `returns`: `dict`
- `side_effect`: none
- `leakage_level`: L1
- `limitations`: Current task only.
- `backend_source`: harness memory

## `check_activity_success`

- `signature`: `check_activity_success()`
- `canonical_family`: `VERIFY`
- `description`: Return safe success/completion/score for the current symbolic task.
- `arguments`: `{}`
- `returns`: `dict`
- `side_effect`: none
- `leakage_level`: L1 verifier summary
- `limitations`: Does not reveal the expected plan.
- `backend_source`: local symbolic predicate verifier
