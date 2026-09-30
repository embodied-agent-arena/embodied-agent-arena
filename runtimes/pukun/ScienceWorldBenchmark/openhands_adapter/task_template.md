# ScienceWorld Text Task

You are working in RoBench's `persistent_repl_code` runtime. The same coding
conversation, Python globals, ScienceWorld backend, and episode persist.

## Task

- Benchmark/track: {benchmark} / {track}
- Suite/instance: {suite} / {instance_id}
- Task: {task_id} / {task_name}
- Split/variation/simplification: {source_split} / {variation_idx} / {simplification}
- Goal: {goal_text}
- Budgets: {max_env_steps} environment steps; {max_primitive_calls} primitive calls

Adapter mode: {mode_instructions}

## Code-cell Contract

Write only the next natural code cell to `cell.py`; the harness executes it.
On later turns, read `turn_feedback_latest.md` and `cells/`, then continue from
the existing globals and current world state without replaying earlier cells.
Do not execute code or call primitives during authoring.

Use only `primitive_api.py` and `primitive_cards.md`. Do not edit them, create
other Python entrypoints, inspect environment variables/processes/private or
parent files, or access the official engine directly. Refresh observations and
valid actions after state changes, check success frequently, keep cells
bounded, and do not call `exit()`.
