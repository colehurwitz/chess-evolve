"""Pipeline definition: build_position_eval_workflow().

Provides the DataNode-driven workflow used by evolution.py to evaluate
chess positions via SwarmEngine.
"""

from __future__ import annotations

from factory.workflow.primitives import AgentNode, AgentRole, DataNode, Workflow

POSITION_TASK_REF = "chess_evolve.tasks:PositionTask"
GAME_TASK_REF = "chess_evolve.tasks:GameTask"

GENERATOR_PROMPT = (
    "You are playing chess. Look at the board position and legal moves. "
    "Pick a move. Output ONLY the UCI move (e.g. e2e4). Nothing else."
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
    generator agent picks a move, then ``game_gate`` advances the game state
    (applies the move, plays Stockfish's response, updates board files).
    The loop continues until the game is over.  ``GameTask.verify()`` then
    scores the completed game.
    """
    from chess_evolve.config import MAX_MOVES

    if cfg is None:
        cfg = PipelineConfig()

    # Reuse the generator AgentNode from the base pipeline
    base_wf = build_pipeline(cfg).compile()
    generator = base_wf.nodes["generator"].model_copy(deep=True)

    # Game gate: advances game state after each LLM move
    game_gate = GateNode(
        id="game_gate",
        evaluator_type="fn",
        evaluator_command=(
            "python3 -c '"
            "from chess_evolve.engine import advance_game_state; "
            "advance_game_state(\"{project_path}\")"
            "'"
        ),
    )

    # Subgraph: Loop(generator → game_gate), max MAX_MOVES iterations
    generator_pkg = Package(
        name="move-generator",
        inputs=[Port(name="board", artifact_path=".factory/chess/board_state.md")],
        outputs=[Port(name="move", artifact_path=".factory/chess/move.md")],
        graph=Workflow(
            name="move-gen",
            nodes={"generator": generator},
            edges=[],
            start_node="generator",
        ),
        entry_node="generator",
        exit_node="generator",
    )
    game_body = Sequential(generator_pkg, name="generate-move")
    game_loop = Loop(
        game_body,
        game_gate,
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
        subgraph_exit="game_gate",
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
