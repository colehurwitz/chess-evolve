"""Pipeline definition: build_position_eval_workflow().

Provides the DataNode-driven workflow used by evolution.py to evaluate
chess positions via SwarmEngine.
"""

from __future__ import annotations

from dataclasses import dataclass

from factory.workflow.primitives import (
    AgentNode,
    AgentRole,
    DataNode,
    Edge,
    GateNode,
    Workflow,
)


@dataclass
class PipelineConfig:
    """Minimal config for pipeline construction."""

    opponent_elo: int = 1500
    max_retries: int = 3
    board_representation: str = "fen"
    use_game_context: bool = True

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


BUILDER_PROMPT = (
    "Read the chess move from .factory/strategy/research.md. "
    "Apply it to the game state in .factory/chess/game_state.json: "
    "push the move UCI string to move_list, increment move_count, "
    "and update the FEN using python-chess. "
    "Then update .factory/chess/board_state.md with the new board "
    "position, legal moves for the side to move, and the correct "
    "'You are playing …' label. "
    "Finally update .factory/current_item.json prompt with the "
    "new FEN, legal moves, and board so the next move can be chosen."
)


def build_game_eval_workflow(cfg: PipelineConfig | None = None) -> Workflow:
    """Build a DataNode-driven workflow for full-game evaluation.

    The subgraph uses **researcher → builder → gate_qa** to match the
    node IDs that ``factory.outer_loop.designer.design_minimal`` creates.
    By freezing ``builder`` in ``frozen_node_ids``, the designer cannot
    replace it with an empty-prompt AgentNode — the chess-specific
    prompt template is preserved across generations.

    A ``DataNode`` (``task_ref=GameTask``) iterates over game configs
    (ELO × color).  For each game instance the subgraph runs:

    1. **researcher** — picks a move (AgentNode, gets board via
       ``initial_context``)
    2. **builder** — applies the move to game state files (AgentNode
       with chess-specific ``prompt_template``)
    3. **gate_qa** — reviews the result and RELOOPs if needed

    ``GameTask.verify()`` scores the completed game.
    """
    if cfg is None:
        cfg = PipelineConfig()

    researcher = AgentNode(
        id="researcher",
        role=AgentRole.RESEARCHER,
        prompt_template=GENERATOR_PROMPT,
        reads={".factory/chess/board_state.md", ".factory/chess/memory.md"},
        writes={".factory/strategy/research.md"},
        timeout=300,
    )

    builder = AgentNode(
        id="builder",
        role=AgentRole.BUILDER,
        prompt_template=BUILDER_PROMPT,
        reads={
            ".factory/strategy/research.md",
            ".factory/chess/game_state.json",
        },
        writes={
            ".factory/reviews/builder-latest.md",
            ".factory/chess/game_state.json",
            ".factory/chess/board_state.md",
        },
        timeout=600,
    )

    gate_qa = GateNode(
        id="gate_qa",
        evaluator_type="agent",
        evaluator_role=AgentRole.CEO,
        gate_prompt=(
            "Read the builder output at .factory/reviews/builder-latest.md. "
            "Verify the game state was updated correctly (FEN, move_list, "
            "board_state). PROCEED if correct, RELOOP to builder if not."
        ),
        reads={".factory/reviews/builder-latest.md"},
    )

    data_node = DataNode(
        id="games",
        task_ref=GAME_TASK_REF,
        subgraph_entry="researcher",
        subgraph_exit="gate_qa",
        parallelism=1,
        writes={".factory/chess/game_results.json"},
    )

    nodes = {
        "games": data_node,
        "researcher": researcher,
        "builder": builder,
        "gate_qa": gate_qa,
    }
    edges = [
        Edge(source="researcher", target="builder"),
        Edge(source="builder", target="gate_qa"),
    ]

    return Workflow(
        name="game-eval",
        nodes=nodes,  # type: ignore[arg-type]
        edges=edges,
        start_node="games",
    )
