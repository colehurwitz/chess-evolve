# Builder Review — OptKnob additions to chess-game workflow

**Date:** 2026-09-14
**Branch:** factory/run-df98cb97
**Status:** ✅ COMPLETE

## Changes Made

### `.factory/workflows/chess_game.py`
- Added `OptKnob` to imports from `factory.workflow.package`
- Removed unused `GENERATOR_PROMPT` import (prompt now built locally)
- Added `_VERBOSITY_INSTRUCTIONS` dict mapping verbosity levels to instruction strings
- Added `build_generator_prompt(style, verbosity)` function for constructing prompts at workflow build time
- Defined `style_knob` (expandable, 4 bounds) and `verbosity_knob` (non-expandable, 3 bounds) inside `workflow()`
- Updated `AgentNode` to use `build_generator_prompt()` with knob defaults
- Added `knobs=[style_knob, verbosity_knob]` to the `Package` constructor
- Propagated `knob_values`, `knob_bounds`, `knob_expandable` from `loop_wf` to the final `Workflow` return (needed since the assemble step creates a new Workflow object)
- Renumbered section comments (1→knobs, 2→generator, 3→gate, 4→package, 5→loop, 6→compile, 7→datanode, 8→assemble)

### `tests/test_chess_game_workflow.py`
- Added `importlib.util` import
- Added `_load_chess_game_module()` helper function
- Added `TestBuildGeneratorPrompt` class (5 tests) — validates prompt builder with various style/verbosity combos and fallback behavior
- Added `TestOptKnobs` class (6 tests) — validates knob_values, knob_bounds, knob_expandable propagation and prompt content
- Added `TestKnobMutateIntegration` class (3 tests) — validates KNOB_MUTATE works end-to-end
- All 10 existing tests remain unchanged and pass

## Implementation Note

The spec's target state didn't include `knob_values`/`knob_bounds`/`knob_expandable` in the final `Workflow(...)` constructor (section 8). This was needed because `Package.compile()` sets these on the inner graph, but the assemble step creates a new `Workflow` that drops them. Added the three kwargs to propagate knob metadata through to the final workflow.

## Test Results
- 24/24 tests pass (10 existing + 14 new)
- ruff: all checks passed
- No changes to pipeline.py, evolution.py, or chess_game_gate.py
