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


def build_game_eval_workflow() -> Workflow:
    """Build a DataNode-driven workflow for full-game evaluation.

    Uses ``researcher`` / ``builder`` / ``gate_qa`` node IDs that match the
    names produced by :pymethod:`DesignerAgent.design_minimal` so that the
    SwarmEngine designer preserves chess-specific prompts on frozen nodes
    instead of replacing them with empty-prompt generic ones.

    Subgraph flow per game instance::

        researcher → builder → gate_qa

    * **researcher** — reads the board and writes strategic analysis.
    * **builder** — reads analysis + board, outputs a UCI move.  Carries a
      chess-specific ``prompt_template`` so the agent knows to emit a move.
    * **gate_qa** — a ``GateNode`` (``evaluator_type='fn'``) that calls
      ``advance_game_state`` to apply the move, play Stockfish's reply, and
      update board files.  Prints ``RELOOP`` (continue) or ``PROCEED``
      (game over).

    ``GameTask.verify()`` scores the completed game from
    ``game_state.json``.
    """
    from factory.workflow.primitives import Edge, GateNode

    researcher = AgentNode(
        id="researcher",
        role=AgentRole.RESEARCHER,
        reads={".factory/chess/board_state.md", ".factory/chess/memory.md"},
        writes={".factory/strategy/research.md"},
        prompt_template=(
            "Analyze the chess position in board_state.md. "
            "Suggest candidate moves with brief reasoning."
        ),
    )

    builder = AgentNode(
        id="builder",
        role=AgentRole.BUILDER,
        reads={".factory/strategy/research.md", ".factory/chess/board_state.md"},
        writes={".factory/chess/move.md"},
        prompt_template=GENERATOR_PROMPT,
    )

    gate_qa = GateNode(
        id="gate_qa",
        evaluator_type="fn",
        evaluator_command=(
            "python3 -c '"
            "from chess_evolve.engine import advance_game_state; "
            "advance_game_state(\"{project_path}\")"
            "'"
        ),
        reads={".factory/chess/move.md"},
    )

    data_node = DataNode(
        id="games",
        task_ref=GAME_TASK_REF,
        subgraph_entry="researcher",
        subgraph_exit="gate_qa",
        parallelism=1,
        writes={".factory/chess/game_results.json"},
    )

    return Workflow(
        name="game-eval",
        nodes={
            "games": data_node,
            "researcher": researcher,
            "builder": builder,
            "gate_qa": gate_qa,
        },
        edges=[
            Edge(source="researcher", target="builder"),
            Edge(source="builder", target="gate_qa"),
        ],
        start_node="games",
    )
