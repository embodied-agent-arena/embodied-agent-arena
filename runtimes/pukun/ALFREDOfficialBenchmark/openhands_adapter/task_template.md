# ALFRED Official Visual Task

You are working in RoBench's `persistent_repl_code` runtime. One coding-agent
conversation, Python interpreter, ALFRED environment, and episode persist.

## Task

- Benchmark/track: {benchmark} / {track}
- Suite/instance: {suite} / {instance_id}
- Task: {task_id} ({task_type}, {source_split})
- Scene: {scene}
- Goal: {goal_text}
- Environment-step budget: {max_env_steps}
- Boundary: {boundary}

Official steps, if supplied:

{step_by_step_instructions}

Mode guidance: {run_mode_guidance}

## Code-cell Contract

Write only the next natural code cell to `cell.py`; the harness executes it.
On later turns read `turn_feedback_latest.md` and `cells/`, then continue from
existing globals and the current episode. Do not replay earlier cells and do
not execute code or primitives during authoring.

Use only `primitive_api.py` according to `primitive_cards.md`. Do not edit the
facade/cards, create alternate entrypoints, inspect environment variables or
private/parent files, access THOR directly, or read hidden trajectory fields.
Ground objects through public visual primitives, inspect each action result,
and call the verifier. Keep each cell bounded and do not call `exit()`.
