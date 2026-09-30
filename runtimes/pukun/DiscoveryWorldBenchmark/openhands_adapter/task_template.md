# DiscoveryWorld Task

You are working in RoBench's `persistent_repl_code` runtime. The same coding
conversation, Python interpreter, DiscoveryWorld backend, and episode persist.

## Task

- Benchmark/track: {benchmark} / {track}
- Suite/instance: {suite} / {instance_id}
- Task: {task_id} ({task_type})
- Scenario/difficulty/seed: {scenario_name} / {difficulty} / {seed}
- Goal: {goal_text}
- Environment-step budget: {max_env_steps}

## Code-cell Contract

Write only the next natural code cell to `cell.py`; the harness executes it.
On later turns read `turn_feedback_latest.md` and `cells/`, then continue from
persistent globals and the current episode without replaying earlier cells.
Do not execute code or primitives during authoring.

Use only `primitive_api.py` according to `primitive_cards.md`. Do not edit the
facade/cards, create alternate Python entrypoints, inspect environment variables
or process/private/parent files, access DiscoveryWorld directly, or read hidden
state. Validate action calls, inspect compact observations/evidence, check the
verifier regularly, keep every cell bounded, and do not call `exit()`.
