# chess-evolve

[![CI](https://github.com/lambdabaa/chess-evolve/actions/workflows/ci.yml/badge.svg)](https://github.com/lambdabaa/chess-evolve/actions/workflows/ci.yml)

Teach an LLM to play chess via evolutionary prompt optimization, powered by [remote-factory](https://github.com/akashgit/remote-factory).

![Live dashboard showing score trends, mutation distribution, and parallel games](docs/screenshot.png)

Each chess move runs a composed `Package` pipeline through factory's `WorkflowExecutor`:

```
Sequential(
    Parallel(analyst, tactician, positionalist),
    Loop(Sequential(selector, verifier), gate)
)
```

The outer loop evolves this pipeline using factory's MAP-Elites quality-diversity search:
- `KNOB_MUTATE` tunes verification style, iteration count, and critique approach
- `PROMPT_MUTATE` rewrites agent prompts via Opus (full replacement, not append)
- Contrastive reflection identifies which knobs and prompts drive performance
- Rank-weighted tournament selection biases toward stronger parents

## Quick start

```bash
uv sync
brew install stockfish  # macOS (or apt install stockfish)

# Run the evolution loop
chess-evolve run

# In another terminal, start the live UI
chess-evolve serve
# Open http://localhost:8422
```

## Evaluating move quality (`eval-positions`)

Score the move generator on a fixed set of positions instead of playing full
games. This runs a DataNode-driven batch evaluation over a list of FEN
positions: for each position the LLM picks a move, and it is scored by
**centipawn loss** — how much worse its move is than Stockfish's best move at
the same position.

```bash
uv run chess-evolve eval-positions \
    [--positions eval/test_positions.json] \
    [--depth N] \
    [--time SECONDS]
```

- `--positions` — path to the positions JSON file (default `eval/test_positions.json`).
- `--depth N` — Stockfish analysis depth (default 12 when neither `--depth` nor `--time` is set).
- `--time SECONDS` — Stockfish analysis time per position, in seconds. If set, it takes precedence over `--depth`.

The command prints one line per position (chosen move, Stockfish's best move,
centipawn loss, score, pass/fail) followed by a summary with **mean centipawn
loss**, **mean score**, and **pass rate**. It also writes the full aggregate to
`.factory/chess/eval_results.json` in the workspace.

Stockfish is located via the `STOCKFISH_PATH` environment variable; if unset it
falls back to a `PATH` lookup (`stockfish`) and then a few common install
locations. Live runs require Stockfish plus an LLM backend for move generation.

Sample output:

```
$ uv run chess-evolve eval-positions --depth 12
pos-001: move=e2e4 best=e2e4 cpl=0 score=1.000 pass=True
pos-002: move=g1f3 best=d2d4 cpl=35 score=0.650 pass=True
pos-003: move=f1c4 best=e1g1 cpl=120 score=0.000 pass=False

mean_cpl=51.7 mean_score=0.550 pass_rate=0.67 (2/3)
results written to /tmp/chess-factory/position-eval/.factory/chess/eval_results.json
```

## Adapting this to your domain

This demo has two layers: **factory integration** (reusable pattern) and **chess logic** (domain-specific). If you're building something similar for a different domain, here's what to keep, what to replace, and what to study.

### The factory integration pattern (study these)

These files show the pattern you'd follow regardless of domain:

**[`pipeline.py`](src/chess_evolve/pipeline.py)** defines the workflow as a composition of Packages:
```python
pipeline = Sequential(
    Parallel(analyst_pkg, tactical_pkg, positional_pkg),
    Loop(Sequential(selector_pkg, verifier_pkg), gate, max_iterations=2),
)
wf = pipeline.compile()  # lowers to flat DAG with knob_values
```
Your version: compose your domain's agents into a pipeline using `Sequential`, `Parallel`, `Loop`, and `Conditional`.

**[`pipeline.py`](src/chess_evolve/pipeline.py)** also declares `OptKnob`s on each Package, telling factory what it's allowed to mutate:
```python
OptKnob(name="verify_style", kind="prompt", node_id="verifier",
        default="strict", bounds=["strict", "standard", "lenient"],
        expandable=True, expansion_hint="Verification approach")
```
Your version: declare knobs for your domain's tunable parameters.

**[`evolution.py`](src/chess_evolve/evolution.py)** runs the outer loop as a thin `SwarmEngine` consumer:
```python
task = PositionTask()
workflow = build_position_eval_workflow()

config = SwarmConfig(
    benchmark="chess-evolve",
    budget=100,
    frozen_node_ids=["positions"],
)
config.set_task(task)

evaluator = SwarmEvaluator(config, inner_loop_factory=True, project_dir=project_dir)
engine = SwarmEngine(config, evaluator, project_dir=project_dir)
result = engine.run(workflow, project_dir=str(project_dir))
```
Your version: define a `Task`, build a workflow, and hand both to `SwarmEngine`. The engine handles mutation, selection, diversity preservation, and reflection internally.

Factory's `compute_features()` automatically extracts MAP-Elites feature dimensions from the compiled workflow (knob values, prompt content, edge structure, params). No custom feature function needed.

### Domain-specific code (replace these)

**[`prompts.py`](src/chess_evolve/prompts.py)** contains all chess-specific prompt templates. Your version: write prompts for your domain's agents.

**[`engine.py`](src/chess_evolve/engine.py)** handles the LLM-to-domain interface: sending board state to the LLM, parsing moves from its output, calling Stockfish for the opponent. Your version: implement your domain's I/O (e.g., sending a code problem to the LLM, parsing its solution, running tests).

**[`game.py`](src/chess_evolve/game.py)** plays a single game and computes a score. `EvalResult` holds the outcome; `play_game()` manages the game loop; `evaluate_pipeline()` runs N games and averages. Your version: implement your domain's evaluation (run the task, measure quality, return a score).

### What factory provides (you don't write these)

| Component | What it does |
|---|---|
| `Package`, `Sequential`, `Parallel`, `Loop`, `Conditional` | Compose agents into a DAG |
| `OptKnob`, `StateContract`, `Port` | Declare the optimization surface |
| `WorkflowExecutor` | Execute the compiled DAG |
| `MAPElitesArchive` | Quality-diversity archive (preserves diverse strategies) |
| `OuterLoopReflector` | Contrastive analysis (what works, what doesn't, why) |
| `apply_random_mutation` | Mutation operators: `KNOB_MUTATE`, `PROMPT_MUTATE`, `PARAM_MUTATE` |
| `WeightedRandomStrategy` | Configurable operator selection weights |
| `CycleRecord` | Structured experiment data for reflection |
| `default_prompt_rewriter` | Opus-powered full prompt rewriting |
| `default_knob_expander` | Opus-powered knob value invention |
| `Population.make_individual` | Creates an Individual with auto-computed features |
| `compute_features` | Extracts MAP-Elites features from a compiled workflow |

#### Integration details

**WorkflowExecutor** runs the compiled DAG but uses `factory.agents.runner.invoke_agent` to call agents. To use your own LLM backend, swap the module attribute before execution:
```python
import factory.agents.runner as runner
runner.invoke_agent = my_custom_invoke  # (role, task, path, **kw) -> (text, code)
executor = WorkflowExecutor(workflow=wf, project_path=workspace)
result = await executor.execute()
move = result.node_outputs["selector"]  # read outputs from memory
```

**Prompt mutation carry-over**: `apply_random_mutation` stores rewritten prompts in `_prompt_*` knob values on the workflow IR. When rebuilding a Package from config, copy these through so `compile()` preserves them:
```python
child_wf, rec = apply_random_mutation(parent_wf, strategy, gen)
child_pipeline = build_pipeline(child_cfg)
for k, v in child_wf.knob_values.items():
    if k.startswith("_prompt_"):
        child_pipeline.graph.knob_values[k] = v
        child_pipeline.graph.knob_expandable[k] = child_wf.knob_expandable.get(k, "")
```

**`default_prompt_rewriter`** is called automatically by `PROMPT_MUTATE`. It spawns Opus via CLI to rewrite an agent's prompt based on reflection hints. No setup needed -- it's the default parameter on `mutate_prompt()`.

**`default_knob_expander`** is called when all knob bounds are exhausted and the knob is marked `expandable=True`. It spawns Opus to invent new values (e.g., a new prompt variant). Also automatic.

### Wiring the feedback loop

The outer loop needs structured feedback to learn. Here's how chess-evolve connects evaluation results back to factory's reflector:

**Build a `CycleRecord`** from your evaluation results ([`game.py:to_cycle_record()`](src/chess_evolve/game.py)). Include domain-specific context in the `ExperimentRecord.hypothesis` field -- this is what the reflector reads during contrastive analysis. Chess-evolve includes move history, eval curve, blunder locations, selector reasoning, and verifier output.

**Prompt mutations are handled automatically** by `SwarmEngine`. The engine's internal reflector sees both knob values and `_prompt_<node>` entries on the workflow IR — no manual wiring needed in your consumer code.

**Separate format from strategy in prompts.** If your pipeline has agents with strict output format requirements (e.g., "respond with exactly one UCI move"), put the format constraint in the system prompt (`_sdk_invoke_agent`), not the `prompt_template`. The `PROMPT_MUTATE` operator rewrites `prompt_template` -- if format rules are mixed in, the rewriter can break the output contract.

**Read agent outputs from memory**, not the filesystem. `result.node_outputs["selector"]` gives you the agent's response directly. The filesystem path (`move.md`, `reviews/strategist-latest.md`) is unreliable -- files can be stale, missing, or from a previous pipeline run.

**Track failure modes** so the reflector can learn from them. Chess-evolve logs illegal moves (selector produced valid notation but the move isn't legal on the current board) and includes them in the CycleRecord so the reflector can steer toward configs that produce fewer hallucinated moves.

**Use two-tier reflection.** Factory's `OuterLoopReflector` does fast structural comparison (knob gradients, step counts) which drives guided mutations. But for domain-aware insights, chess-evolve also runs an LLM reflection pass (Opus via CLI) that reads actual game data and produces coaching advice like "stop relocating knights that are defending pieces." The structural reflector feeds `reflection_report` to `apply_random_mutation` for guided knob selection; the LLM reflection feeds `prompt_improvements` for prompt rewriting hints. Both run each generation — structural for mutation guidance, LLM for deep analysis.

### The minimal integration

At its simplest, using factory as a library requires three things:

1. **Define your pipeline** with `Package` + composition operators
2. **Declare `OptKnob`s** on the parameters you want optimized
3. **Write an `evaluate()` function** that takes a compiled workflow and returns a score

Factory handles mutation, selection, diversity preservation, and reflection automatically.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `CHESS_MODEL` | `opus` | Model for the `claude` CLI (move generation uses Vertex Haiku) |
| `CHESS_WORKSPACE` | `/tmp/chess-factory` | Working directory for game data |
| `STOCKFISH_PATH` | (auto) | Path to the Stockfish binary; falls back to `PATH` lookup then common install locations |

