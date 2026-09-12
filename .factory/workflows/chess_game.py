"""chess-game workflow — DataNode-driven full-game evaluation against Stockfish.

Project-local workflow discovered automatically by the factory registry.
Simple: one generator picks a move, one gate advances the game.
The outer loop evolves prompt complexity.
"""

from __future__ import annotations

import sys

from chess_evolve.config import MAX_MOVES
from factory.workflow.package import Loop, Package, Port, Sequential
from factory.workflow.primitives import (
    AgentNode,
    AgentRole,
    DataNode,
    GateNode,
    Workflow,
)

GAME_TASK_REF = "chess_evolve.tasks:GameTask"

GENERATOR_PROMPT = (
    "You are playing chess. Look at the board position and legal moves. "
    "Pick a move. Output ONLY the UCI move (e.g. e2e4). Nothing else."
)

meta = {
    "name": "chess-game",
    "description": "DataNode-driven full-game evaluation against Stockfish",
}


def workflow() -> Workflow:
    """Simple game loop: generator picks move, gate advances game."""
    generator = AgentNode(
        id="generator",
        role=AgentRole.STRATEGIST,
        prompt_template=GENERATOR_PROMPT,
        reads={".factory/chess/board_state.md", ".factory/chess/memory.md"},
        writes={".factory/chess/move.md"},
    )

    game_gate = GateNode(
        id="game_gate",
        evaluator_type="fn",
        evaluator_command=(
            f"{sys.executable} -c '"
            "from chess_evolve.engine import advance_game_state; "
            "advance_game_state(\"{project_path}\")"
            "'"
        ),
    )

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

    game_loop = Loop(
        Sequential(generator_pkg, name="generate-move"),
        game_gate,
        max_iterations=MAX_MOVES,
        name="game-loop",
    )

    loop_wf = game_loop.compile()

    data_node = DataNode(
        id="games",
        task_ref=GAME_TASK_REF,
        subgraph_entry=loop_wf.start_node,
        subgraph_exit="exit_game-loop",
        parallelism=1,
        writes={".factory/chess/game_results.json"},
    )

    return Workflow(
        name="chess-game",
        nodes={"games": data_node, **loop_wf.nodes},
        edges=loop_wf.edges,
        start_node="games",
    )
