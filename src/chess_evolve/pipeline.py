"""Pipeline definition: build_position_eval_workflow().

Provides the DataNode-driven workflow used by evolution.py to evaluate
chess positions via SwarmEngine.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from factory.workflow.package import Loop, Package, Port, Sequential
from factory.workflow.primitives import AgentNode, AgentRole, DataNode, Edge, GateNode, Workflow


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

RESEARCHER_PROMPT = (
    "You are a chess analyst. Study the board position, evaluate candidate moves, "
    "and recommend the best move. Explain your reasoning briefly, then state "
    "your recommended move in UCI format (e.g. e2e4) on the last line."
)

BUILDER_PROMPT = (
    "You are a chess move selector. Read the board position and the analyst's "
    "recommendation. Pick the best legal move.\n\n"
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
    """DEPRECATED: Use .factory/workflows/chess_game.py via WorkflowRegistry instead.

    Build a DataNode-driven workflow for full-game evaluation.

    Uses ``researcher`` / ``builder`` / ``gate_qa`` node IDs to match the
    designer's naming convention.  The ``researcher`` analyses the position,
    the ``builder`` selects the final UCI move, and ``gate_qa`` advances the
    game state (applies the move, plays Stockfish's response, updates board
    files).  The loop continues until the game is over.
    ``GameTask.verify()`` then scores the completed game.
    """
    import sys
    import warnings

    warnings.warn(
        "build_game_eval_workflow() is deprecated. "
        "Use WorkflowRegistry.get_workflow('chess-game', project_dir) instead. "
        "See .factory/workflows/chess_game.py.",
        DeprecationWarning,
        stacklevel=2,
    )

    from chess_evolve.config import MAX_MOVES

    if cfg is None:
        cfg = PipelineConfig()

    # Researcher: analyses the position and recommends a move
    researcher = AgentNode(
        id="researcher",
        role=AgentRole.RESEARCHER,
        prompt_template=RESEARCHER_PROMPT,
        reads={".factory/chess/board_state.md", ".factory/chess/memory.md"},
        writes={".factory/chess/analysis.md"},
    )

    # Builder: reads analysis and board, outputs the final UCI move
    builder = AgentNode(
        id="builder",
        role=AgentRole.BUILDER,
        prompt_template=BUILDER_PROMPT,
        reads={
            ".factory/chess/board_state.md",
            ".factory/chess/analysis.md",
            ".factory/chess/memory.md",
        },
        writes={".factory/chess/move.md"},
    )

    # Gate QA: advances game state after each LLM move
    gate_qa = GateNode(
        id="gate_qa",
        evaluator_type="fn",
        evaluator_command=(
            f"{sys.executable} -c '"
            "from chess_evolve.engine import advance_game_state; "
            "advance_game_state(\"{project_path}\")"
            "'"
        ),
    )

    # Subgraph: researcher → builder inside a Package
    move_pkg = Package(
        name="move-generator",
        inputs=[Port(name="board", artifact_path=".factory/chess/board_state.md")],
        outputs=[Port(name="move", artifact_path=".factory/chess/move.md")],
        graph=Workflow(
            name="move-gen",
            nodes={"researcher": researcher, "builder": builder},
            edges=[Edge(source="researcher", target="builder")],
            start_node="researcher",
        ),
        entry_node="researcher",
        exit_node="builder",
    )
    game_body = Sequential(move_pkg, name="generate-move")
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
