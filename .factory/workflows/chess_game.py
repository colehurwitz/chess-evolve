"""chess-game workflow — DataNode-driven full-game evaluation against Stockfish.

Project-local workflow discovered automatically by the factory registry.
Plays full games via a Loop(generator → game_gate) pattern, with GameTask
driving per-instance lifecycle (setup, verify, scoring).
"""

from __future__ import annotations

from factory.workflow.package import Loop, Package, Port, Sequential
from factory.workflow.primitives import (
    AgentNode,
    AgentRole,
    DataNode,
    GateNode,
    Workflow,
)

from chess_evolve.config import MAX_MOVES
from chess_evolve.pipeline import GAME_TASK_REF, GENERATOR_PROMPT

meta = {
    "name": "chess-game",
    "description": "DataNode-driven full-game evaluation against Stockfish",
}


def workflow() -> Workflow:
    """Build a DataNode-driven workflow for full-game evaluation.

    Structure::

        DataNode(games)
          └─ subgraph: Loop(generator → game_gate, max_iterations=MAX_MOVES)
               ├─ generator: AgentNode(reads board_state + memory, writes move)
               └─ game_gate: GateNode(advance_game_state → RELOOP | PROCEED)

    Returns:
        Workflow with DataNode 'games' as start node wrapping the game loop.
    """
    # ── 1. Generator AgentNode ──────────────────────────────────
    generator = AgentNode(
        id="generator",
        role=AgentRole.STRATEGIST,
        prompt_template=GENERATOR_PROMPT,
        reads={".factory/chess/board_state.md", ".factory/chess/memory.md"},
        writes={".factory/chess/move.md"},
    )

    # ── 2. Game gate GateNode ───────────────────────────────────
    game_gate = GateNode(
        id="game_gate",
        evaluator_type="fn",
        evaluator_command=(
            "python3 {project_path}/.factory/workflows/chess_game_gate.py"
            ' "{project_path}"'
        ),
    )

    # ── 3. Package the generator for Loop composition ───────────
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

    # ── 4. Loop: generator → game_gate ──────────────────────────
    game_body = Sequential(generator_pkg, name="generate-move")
    game_loop = Loop(
        game_body,
        game_gate,
        max_iterations=MAX_MOVES,
        name="game-loop",
    )

    # ── 5. Compile loop into flat workflow ───────────────────────
    loop_wf = game_loop.compile()

    # ── 6. DataNode wraps the compiled loop subgraph ────────────
    data_node = DataNode(
        id="games",
        task_ref=GAME_TASK_REF,
        subgraph_entry=loop_wf.start_node,
        subgraph_exit="exit_game-loop",
        parallelism=1,
        writes={".factory/chess/game_results.json"},
    )

    # ── 7. Assemble final workflow ──────────────────────────────
    all_nodes: dict = {"games": data_node, **loop_wf.nodes}
    return Workflow(
        name="chess-game",
        nodes=all_nodes,
        edges=loop_wf.edges,
        start_node="games",
    )
