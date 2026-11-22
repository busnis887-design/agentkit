"""CLI demo: python -m agentkit.cli "Calculate 12 * 8 + 4" """

from __future__ import annotations

import argparse

from .agent import Agent
from .tools import create_default_registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentkit")
    parser.add_argument(
        "goal",
        nargs="?",
        default="Calculate 12 * 8 + 4 and explain RAG",
        help="what the agent should do",
    )
    parser.add_argument("--max-steps", type=int, default=5)
    args = parser.parse_args(argv)

    registry = create_default_registry()
    print("Registered tools: " + ", ".join(registry.names()))
    trace = Agent(registry=registry).run(args.goal, max_steps=args.max_steps)
    for line in trace.as_lines():
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
