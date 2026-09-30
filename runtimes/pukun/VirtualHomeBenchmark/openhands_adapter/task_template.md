# VirtualHome Symbolic Task

You are working in RoBench's `persistent_repl_code` runtime. The coding-agent
conversation, Python interpreter, symbolic graph backend, and episode persist.

## Task

- Benchmark/track: {benchmark} / {track}
- Suite/instance: {suite} / {instance_id}
- Task: {task_id} ({task_type})
- Graph: {graph_name}
- Goal: {goal_text}
- Program-step budget: {max_env_steps}

## Code-cell Contract

Write only the next natural code cell to `cell.py`; the harness executes it.
On later turns read `turn_feedback_latest.md` and `cells/`, then continue from
existing globals and the current symbolic episode without replaying prior
cells. Do not execute code or primitives during authoring.

Use only `primitive_api.py` and `primitive_cards.md`. Do not edit them, create
alternate entrypoints, inspect environment variables/processes/private or
parent files, access VirtualHome directly, or read gold programs. Query the
public symbolic state, validate/execute one justified program step at a time,
check success regularly, keep cells bounded, and do not call `exit()`.
