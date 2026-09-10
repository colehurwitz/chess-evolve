"""Pipeline definition: build_position_eval_workflow().

Provides the DataNode-driven workflow used by evolution.py to evaluate
chess positions via SwarmEngine.
"""

from __future__ import annotations

from factory.workflow.primitives import AgentNode, AgentRole, DataNode, Workflow

POSITION_TASK_REF = "chess_evolve.tasks:PositionTask"

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
