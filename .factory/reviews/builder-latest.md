# Builder Review — chess-game workflow

## Changes Made

### 1. `.factory/workflows/chess_game_gate.py` (NEW — 165 lines)
Standalone gate script that contains `advance_game_state()` and its helpers
(`_play_stockfish_move`, `_eval_and_append`, `_check_game_over`) copied from
`engine.py`. Avoids importing from `chess_evolve.engine` which has a broken
`broadcast` import. Called by the GateNode's `evaluator_command`.

### 2. `.factory/workflows/chess_game.py` (NEW — 97 lines)
Project-local workflow definition with `meta` dict and `workflow()` function.
Builds a DataNode-driven game-eval workflow: `DataNode(games)` wraps a
`Loop(generator → game_gate)`. Uses `GAME_TASK_REF` and `GENERATOR_PROMPT`
from `pipeline.py`.

### 3. `tests/test_chess_game_workflow.py` (NEW — 210 lines)
23 hermetic tests covering:
- `TestMeta` — registry metadata
- `TestWorkflowStructure` — node types, names, configuration
- `TestSubgraphConnectivity` — entry/exit node validity
- `TestGraphValidation` — structural validation
- `TestGeneratorConfiguration` — reads/writes
- `TestComposeIntegration` — `compose(workflow, GameTask)` produces InnerLoop
- `TestGateScript` — gate file existence and content

### 4. `src/chess_evolve/evolution.py` (MODIFIED)
Updated `game` branch to load workflow from project-local file via
`importlib.util` instead of `build_game_eval_workflow()`.

## Test Results
- 23/23 new tests pass
- 82/88 total tests pass (6 pre-existing failures in test_game_task.py due to
  broken broadcast import — unchanged by this PR)

## Key Decisions
- Gate script uses `chess_game_gate.py` filename (not inline python) to avoid
  the broken `chess_evolve.engine` import
- GateNode `evaluator_command` references the script file with `{project_path}`
  template variable for worktree portability
- Workflow name is `"chess-game"` (hyphenated) matching factory registry conventions
