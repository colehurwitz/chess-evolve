"""Pipeline definition: build_position_eval_workflow().

Provides the DataNode-driven workflow used by evolution.py to evaluate
chess positions via SwarmEngine.
"""

from __future__ import annotations

from dataclasses import dataclass

from factory.workflow.package import Loop, Package, Port, Sequential
from factory.workflow.primitives import AgentNode, AgentRole, DataNode, GateNode, Workflow


@dataclass
class PipelineConfig:
    """Minimal config for pipeline construction."""
    opponent_elo: int = 1500
    max_retries: int = 3

POSITION_TASK_REF = "chess_evolve.tasks:PositionTask"
GAME_TASK_REF = "chess_evolve.tasks:GameTask"

GENERATOR_PROMPT = (
    "You are playing chess. Look at the board position and legal moves. "
    "Pick a move. Output ONLY the UCI move (e.g. e2e4). Nothing else."
)

BUILDER_PROMPT = (
    "You are a chess move builder. Read the board state and analysis, "
    "then pick the single best legal move. "
    "Output ONLY the UCI move (e.g. e2e4). Nothing else."
)


def build_position_eval_workflow() -> Workflow:
    """Build a DataNode-driven workflow that evaluates positions per-item.

    A ``DataNode`` (``task_ref='chess_evolve.tasks:PositionTask'``) resolves one
    instance per position and runs the move-generator ``AgentNode`` as its
    subgraph — the generator executes once per position.  Per-position
    verification (centipawn loss) is performed by ``PositionTask.verify``.

    ``parallelism=1`` serializes items because every instance shares the same
    ``project_path``; serializing prevents ``board_state.md`` / ``move.md`` from
    racing across positions.
    """
    generator = AgentNode(
        id="generator",
        role=AgentRole.STRATEGIST,
        prompt_template=GENERATOR_PROMPT,
        reads={".factory/chess/board_state.md", ".factory/chess/memory.md"},
        writes={".factory/chess/move.md"},
    )

    data_node = DataNode(
        id="positions",
        task_ref=POSITION_TASK_REF,
        subgraph_entry="generator",
        subgraph_exit="generator",
        parallelism=1,
        writes={".factory/chess/eval_results.json"},
    )

    return Workflow(
        name="position-eval",
        nodes={"positions": data_node, "generator": generator},
        edges=[],
        start_node="positions",
    )


def build_game_eval_workflow(cfg: PipelineConfig | None = None) -> Workflow:
    """Build a DataNode-driven workflow for full-game evaluation.

    A ``DataNode`` (``task_ref=GameTask``) iterates over game configs (ELO ×
    color). For each game instance, the subgraph runs a ``Loop``: the
    researcher analyses the position, the builder picks the move, then
    ``gate_qa`` advances the game state (applies the move, plays Stockfish's
    response, updates board files).  The loop continues until the game is
    over.  ``GameTask.verify()`` then scores the completed game.

    Node IDs (``researcher``, ``builder``, ``gate_qa``) match the names used
    by the factory designer so that ``frozen_node_ids`` keeps the chess-
    specific prompts intact across mutations.
    """
    from chess_evolve.config import MAX_MOVES

    if cfg is None:
        cfg = PipelineConfig()

    # Researcher: analyses the position (reads board state)
    researcher = AgentNode(
        id="researcher",
        role=AgentRole.RESEARCHER,
        prompt_template=GENERATOR_PROMPT,
        reads={".factory/chess/board_state.md", ".factory/chess/memory.md"},
        writes={".factory/chess/analysis.md"},
    )

    # Builder: picks the actual move using board state and analysis
    builder = AgentNode(
        id="builder",
        role=AgentRole.STRATEGIST,
        prompt_template=BUILDER_PROMPT,
        reads={
            ".factory/chess/board_state.md",
            ".factory/chess/memory.md",
            ".factory/chess/analysis.md",
        },
        writes={".factory/chess/move.md"},
    )

    # Gate: advances game state after each LLM move
    gate_qa = GateNode(
        id="gate_qa",
        evaluator_type="fn",
        evaluator_command=(
            "python3 -c '"
            "from chess_evolve.engine import advance_game_state; "
            "advance_game_state(\"{project_path}\")"
            "'"
        ),
    )

    # Subgraph: Loop(Sequential(researcher → builder) → gate_qa)
    researcher_pkg = Package(
        name="position-analysis",
        inputs=[Port(name="board", artifact_path=".factory/chess/board_state.md")],
        outputs=[Port(name="analysis", artifact_path=".factory/chess/analysis.md")],
        graph=Workflow(
            name="analyse",
            nodes={"researcher": researcher},
            edges=[],
            start_node="researcher",
        ),
        entry_node="researcher",
        exit_node="researcher",
    )
    builder_pkg = Package(
        name="move-builder",
        inputs=[Port(name="board", artifact_path=".factory/chess/board_state.md")],
        outputs=[Port(name="move", artifact_path=".factory/chess/move.md")],
        graph=Workflow(
            name="build-move",
            nodes={"builder": builder},
            edges=[],
            start_node="builder",
        ),
        entry_node="builder",
        exit_node="builder",
    )
    game_body = Sequential(researcher_pkg, builder_pkg, name="generate-move")
    game_loop = Loop(
        game_body,
        gate_qa,
        max_iterations=MAX_MOVES,
        name="game-loop",
    )

    # Compile the loop into a flat workflow to extract nodes
    loop_wf = game_loop.compile()

    # DataNode wraps the compiled loop subgraph
    data_node = DataNode(
        id="games",
        task_ref=GAME_TASK_REF,
        subgraph_entry=loop_wf.start_node,
        subgraph_exit="exit_game-loop",
        parallelism=1,
        writes={".factory/chess/game_results.json"},
    )

    all_nodes = {"games": data_node, **loop_wf.nodes}
    return Workflow(
        name="game-eval",
        nodes=all_nodes,
        edges=loop_wf.edges,
        start_node="games",
    )
