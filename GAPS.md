# GAPS — SwarmEngine Consumer Pattern

Gaps discovered when integrating chess-evolve as a pure `(Task, Workflow) → SwarmEngine`
consumer. Each gap is a place where factory's execution layer does not yet fully support
the consumer pattern. These map to [akashgit/remote-factory#1488](https://github.com/akashgit/remote-factory/issues/1488).

## 1. WorkflowExecutor does not honor `node.writes`

`AgentNode.writes` declares which files a node produces (e.g. `.factory/chess/move.md`),
but `WorkflowExecutor` does not actually write agent output to those paths on disk.
`PositionTask.verify()` reads `move.md` from the workspace directory — if it is never
written, `verify()` returns `score=0.0` for every candidate. The consumer cannot work
around this without monkey-patching `invoke_agent`, which violates the consumer contract.

## 2. `_create_worktree` fails in non-git or nested-worktree contexts

`SwarmEngine.run()` calls `_create_worktree()` to isolate each candidate evaluation.
This fails when the project directory is not a git repository root, or when running
inside an existing git worktree (worktree-within-worktree). The consumer passes
`project_dir` explicitly, but the underlying git operations may still fail depending
on the execution environment.

## 3. `invoke_agent` does not support chess-evolve's custom agent backend

chess-evolve uses a custom LLM invocation path (`engine._cli_call_opus`) that is not
compatible with `WorkflowExecutor.invoke_agent()`. The factory executor expects its own
agent dispatch mechanism. There is no extension point for consumers to plug in a custom
agent backend without subclassing or monkey-patching `WorkflowExecutor`.

## 4. DataNode iteration may not propagate task instances through `compose()`

`compose(workflow, task, wt_path)` validates the workflow against the task's
`required_capabilities`, but the DataNode's `iterate()` method may not correctly
propagate the `PositionTask` instance to each position evaluation. The consumer
defines `frozen_node_ids=["positions"]` to protect the DataNode, but the composition
layer may not wire up the task's `positions()` data into the DataNode's iteration
context.

## 5. SwarmEngine event logging replaces broadcast/UI integration

The old evolution loop used `broadcast_eval_result()` and `broadcast_archive()` for
live web UI updates. SwarmEngine logs events to `events.jsonl` but does not provide
a compatible event stream for chess-evolve's Uvicorn-based live UI (`serve.py`).
Consumers have no hook to intercept per-generation events for custom visualization.
