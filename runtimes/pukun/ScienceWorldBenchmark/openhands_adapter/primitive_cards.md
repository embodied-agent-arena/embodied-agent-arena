# ScienceWorld Text Primitive Cards

These are the public primitives exposed to the OpenHands workspace for this adapter entry.
Each card follows the RoBench primitive schema: `name`, `signature`, `canonical_family`, `description`, `arguments`, `returns`, `side_effect`, `leakage_level`, `limitations`, and `backend_source`.

## `get_task_context`

- `signature`: `get_task_context()`
- `canonical_family`: `CTX`
- `description`: Return benchmark, task id/name, task description, variation, simplification, and step budget.
- `arguments`: `{}`
- `returns`: `dict`
- `side_effect`: none
- `leakage_level`: L1/L3
- `limitations`: Does not expose gold paths or hidden simulator state.; OpenHands workspace policy: call this primitive only from solve.py; do not import primitive_api from python -c, stdin Python, heredocs, notebooks, or temporary scripts.
- `backend_source`: TaskRecord + ScienceWorld taskdescription()

## `observe_text_world`

- `signature`: `observe_text_world()`
- `canonical_family`: `CTX`
- `description`: Return the latest ScienceWorld textual observation.
- `arguments`: `{}`
- `returns`: `str`
- `side_effect`: none
- `leakage_level`: L1/L3
- `limitations`: Only returns the currently revealed text observation.; OpenHands workspace policy: call this primitive only from solve.py; do not import primitive_api from python -c, stdin Python, heredocs, notebooks, or temporary scripts.
- `backend_source`: ScienceWorldEnv.step/reset observation cache

## `list_actions`

- `signature`: `list_actions()`
- `canonical_family`: `CTX`
- `description`: Return current grounded valid actions from ScienceWorld info['valid'].
- `arguments`: `{}`
- `returns`: `list[str]`
- `side_effect`: none
- `leakage_level`: L1/L3
- `limitations`: Returns grounded actions, not template actions such as 'go OBJ'.; OpenHands workspace policy: call this primitive only from solve.py; do not import primitive_api from python -c, stdin Python, heredocs, notebooks, or temporary scripts.
- `backend_source`: ScienceWorldEnv get_valid_action_object_combinations via info['valid']

## `inspect_current_state`

- `signature`: `inspect_current_state(query=None, limit=80)`
- `canonical_family`: `STATE`
- `description`: Return a compact observed-only summary of the current room text, inventory text, and current grounded action groups.
- `arguments`: `{"query": "Optional string or list of strings used to surface current actions containing any query term.", "limit": "Maximum actions per group / query-action list."}`
- `returns`: `dict with current_location, visible_entries, exits_or_doors, inventory_entries, query_actions, substance_query_terms, substance_candidates, action_groups, recent_actions, loop_warnings, valid_action_count, boundary`
- `side_effect`: none
- `leakage_level`: L2 observed text + L3 grounded action metadata
- `limitations`: Observation-only helper; it does not move, manipulate, score, plan, or choose an action.; Only summarizes the current public text observation, current inventory text, current valid actions, and actions already attempted by this wrapper.; Substance candidates are parsed from visible text such as 'substance called water' and matched only against current valid actions.; When query=None, substance candidates may be filtered by public goal terms such as water/unknown substance; this is not a hidden solution.; Loop warnings are generic anti-repetition hints from the attempted action trace, not task-specific solutions.; Does not expose hidden objects, hidden room contents, gold paths, future states, or a correct action sequence.; OpenHands workspace policy: call this primitive only from solve.py; do not import primitive_api from python -c, stdin Python, heredocs, notebooks, or temporary scripts.
- `backend_source`: Current observation text + inventory text + ScienceWorld valid action list + RoBench recent action trace

## `filter_actions`

- `signature`: `filter_actions(include=None, exclude=None, startswith=None, limit=80, exclude_failed=True)`
- `canonical_family`: `CTX`
- `description`: Return current grounded valid actions filtered by simple text terms, optionally hiding actions that already failed in this task.
- `arguments`: `{"include": "Optional substring/list of substrings that returned actions must contain; may also be substance_candidate dict(s) from inspect_current_state.", "exclude": "Optional substring or list of substrings to filter out.", "startswith": "Optional action prefix.", "limit": "Maximum returned actions.", "exclude_failed": "When True, suppress actions recorded by step_text_action as repeated invalid/parser-no-match failures."}`
- `returns`: `list[str]`
- `side_effect`: none
- `leakage_level`: L1/L3
- `limitations`: Only filters current valid actions; it does not execute, fuzzy-match, navigate, or choose a plan.; When include is substance_candidate dict(s), it restricts to that candidate's current matching_actions.; Every returned action still must be passed exactly to step_text_action(action).; Use exclude_failed=False only when deliberately rechecking a previously failed action after the state changed.; OpenHands workspace policy: call this primitive only from solve.py; do not import primitive_api from python -c, stdin Python, heredocs, notebooks, or temporary scripts.
- `backend_source`: Current ScienceWorld info['valid'] list

## `list_recent_failures`

- `signature`: `list_recent_failures(limit=20)`
- `canonical_family`: `STATE`
- `description`: Return actions that failed in this task, with failure counts and last error summaries.
- `arguments`: `{"limit": "Maximum failure records to return."}`
- `returns`: `list[dict]`
- `side_effect`: none
- `leakage_level`: L1/L3
- `limitations`: Only reports failures observed through this primitive facade in the current task.; Does not reveal any hidden correct action sequence.; OpenHands workspace policy: call this primitive only from solve.py; do not import primitive_api from python -c, stdin Python, heredocs, notebooks, or temporary scripts.
- `backend_source`: RoBench wrapper failure memory

## `get_score_state`

- `signature`: `get_score_state()`
- `canonical_family`: `VERIFY`
- `description`: Return the current verifier summary, wrapper metrics, and failure-memory count.
- `arguments`: `{}`
- `returns`: `dict`
- `side_effect`: none
- `leakage_level`: L1/L3
- `limitations`: Verifier summary only; it does not expose hidden solutions or future rewards.; OpenHands workspace policy: call this primitive only from solve.py; do not import primitive_api from python -c, stdin Python, heredocs, notebooks, or temporary scripts.
- `backend_source`: ScienceWorld verifier + RoBench wrapper metrics

## `step_text_action`

- `signature`: `step_text_action(action)`
- `canonical_family`: `HACT`
- `description`: Execute exactly one current grounded ScienceWorld action.
- `arguments`: `{"action": "Exact action string copied from list_actions() or filter_actions()."}`
- `returns`: `Compact StepResult dict with valid_action/error/observation_after/verification and bounded action counts/sample.`
- `side_effect`: one native text action
- `leakage_level`: L1/L3
- `limitations`: Action must be copied from list_actions(); no fuzzy matching or planning.; The OpenHands facade returns a compact result; call list_actions() or filter_actions() for the current full valid action set.; Do not print full result/action-list objects into stdout; keep evidence compact.; Some valid ScienceWorld actions, including unrelated 'focus on ...' actions, can immediately fail or terminate a task.; After each side-effecting action, call check_success() and stop if it reports done or a terminal score.; Must be called from python solve.py; ad-hoc python -c snippets and temporary scripts are rejected for side-effecting actions.; After one solve.py process has issued side-effect actions, rerunning solve.py cannot continue manipulating the same backend session.; OpenHands workspace policy: call this primitive only from solve.py; do not import primitive_api from python -c, stdin Python, heredocs, notebooks, or temporary scripts.
- `backend_source`: ScienceWorldEnv.step(action)

## `look`

- `signature`: `look()`
- `canonical_family`: `STATE`
- `description`: Return the current room/world description as a free ScienceWorld action.
- `arguments`: `{}`
- `returns`: `str`
- `side_effect`: none
- `leakage_level`: L1/L3
- `limitations`: Observation-only helper; it does not move, manipulate, or reveal hidden state.; OpenHands workspace policy: call this primitive only from solve.py; do not import primitive_api from python -c, stdin Python, heredocs, notebooks, or temporary scripts.
- `backend_source`: ScienceWorldEnv.look()

## `inventory`

- `signature`: `inventory()`
- `canonical_family`: `STATE`
- `description`: Return the agent inventory as a free ScienceWorld action.
- `arguments`: `{}`
- `returns`: `str`
- `side_effect`: none
- `leakage_level`: L1/L3
- `limitations`: Only reports currently carried objects; it does not expose object locations elsewhere.; OpenHands workspace policy: call this primitive only from solve.py; do not import primitive_api from python -c, stdin Python, heredocs, notebooks, or temporary scripts.
- `backend_source`: ScienceWorldEnv.inventory()

## `write_evidence`

- `signature`: `write_evidence(key, value)`
- `canonical_family`: `EVD`
- `description`: Record agent evidence in trace-local memory.
- `arguments`: `{"key": "Evidence key.", "value": "JSON-serializable value."}`
- `returns`: `dict`
- `side_effect`: none
- `leakage_level`: L1/L3
- `limitations`: Trace-local memory only; it cannot alter the ScienceWorld environment or verifier state.; OpenHands workspace policy: call this primitive only from solve.py; do not import primitive_api from python -c, stdin Python, heredocs, notebooks, or temporary scripts.
- `backend_source`: RoBench trace memory

## `read_evidence`

- `signature`: `read_evidence()`
- `canonical_family`: `EVD`
- `description`: Return evidence previously written by the generated code.
- `arguments`: `{}`
- `returns`: `dict`
- `side_effect`: none
- `leakage_level`: L1/L3
- `limitations`: Returns only evidence previously written by this generated code run.; OpenHands workspace policy: call this primitive only from solve.py; do not import primitive_api from python -c, stdin Python, heredocs, notebooks, or temporary scripts.
- `backend_source`: RoBench trace memory

## `check_success`

- `signature`: `check_success()`
- `canonical_family`: `VERIFY`
- `description`: Return the current native ScienceWorld verifier state.
- `arguments`: `{}`
- `returns`: `dict with success, score, done, reward, source`
- `side_effect`: none
- `leakage_level`: L1/L3
- `limitations`: Returns verifier summary only; it does not reveal a gold action sequence or hidden solution.; OpenHands workspace policy: call this primitive only from solve.py; do not import primitive_api from python -c, stdin Python, heredocs, notebooks, or temporary scripts.
- `backend_source`: ScienceWorldEnv info['score']/done
