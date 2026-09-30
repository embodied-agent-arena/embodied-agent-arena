# ALFWorld Visual Task

You are working in RoBench's `persistent_repl_code` runtime. This coding-agent
conversation, Python interpreter, visual backend, and task episode persist.

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
the persistent globals and current visual episode without replaying earlier
cells. Do not execute code or primitives during authoring.

Use only `primitive_api.py` according to `primitive_cards.md`. Do not edit the
facade/cards, create helper entrypoints, inspect environment variables or
parent/private files, access THOR directly, or read hidden scene/task state.
Use perception primitives to ground visible objects before acting, inspect each
result, and call the verifier. Keep each cell bounded and do not call `exit()`.
