# ALFWorld Text Task

You are working in RoBench's `persistent_repl_code` runtime. This coding-agent
conversation, Python interpreter, and ALFWorld episode persist across turns.

## Task

- Benchmark/track: {benchmark} / {track}
- Suite/instance: {suite} / {instance_id}
- Task: {task_id} ({task_type}, {source_split})
- Goal: {goal_text}
- Environment-step budget: {max_env_steps}

## Code-cell Contract

Write only the next natural code cell to `cell.py`. The harness executes it;
do not run it yourself. On later turns, read `turn_feedback_latest.md` and the
archived cells under `cells/`, then continue without replaying earlier cells.
Imports, globals, functions, and client objects from successful earlier cells
still exist. The benchmark episode is never reset between cells.

Use only `primitive_api.py` and the documented functions in
`primitive_cards.md`. Do not edit those files, create alternate Python
entrypoints, inspect environment variables/processes/parent directories, or
read private manifests, `.env` files, backend code, gold trajectories, or
hidden evaluator state. Primitive calls cannot be made during authoring.

After each action, inspect its result and call `check_success()` before choosing
the next action. Refresh `list_actions()` instead of repeating an action that
made no progress. Keep every cell bounded; do not call `exit()`.

For pick/place, locate and pick the target before navigating to and placing it.
For clean/heat/cool, pick it up, visit the required appliance, apply the state
change, then complete the placement goal. Prefer exact numbered object names
from current admissible actions when a base name is ambiguous.
