"""Agent loop: decide -> call tool -> observe -> repeat."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from .tools import ToolRegistry, create_default_registry


@dataclass(frozen=True)
class Step:
    thought: str
    tool: str
    arguments: dict[str, Any]
    observation: str


@dataclass
class Trace:
    goal: str
    steps: list[Step] = field(default_factory=list)
    final_answer: str = ""

    def as_lines(self) -> list[str]:
        lines = [f"Goal: {self.goal}"]
        for i, step in enumerate(self.steps, 1):
            lines.append(
                f"{i}. {step.thought} -> {step.tool}({step.arguments}) "
                f"=> {step.observation}"
            )
        lines.append(f"Answer: {self.final_answer}")
        return lines


class Policy(Protocol):
    def decide(
        self, goal: str, registry: ToolRegistry, steps: list[Step]
    ) -> tuple[str, dict[str, Any], str] | None:
        ...


class RuleBasedPolicy:
    """Deterministic stand-in for an LLM planner; useful for tests and demos."""

    def decide(
        self, goal: str, registry: ToolRegistry, steps: list[Step]
    ) -> tuple[str, dict[str, Any], str] | None:
        used = {step.tool for step in steps}
        lowered = goal.lower()

        if "calculat" in lowered and "calculator" not in used:
            match = re.search(r"[-+]?\d[\d\s+*/().-]*", goal)
            expression = match.group(0).strip() if match else ""
            if expression:
                return "calculator", {"expression": expression}, "Need to compute the expression."

        if ("word" in lowered or "character" in lowered or "stat" in lowered) and "text_stats" not in used:
            match = re.search(r"[\"']([^\"']+)[\"']", goal)
            text = match.group(1) if match else goal
            return "text_stats", {"text": text}, "Need text statistics."

        if "json" in lowered and "json_validate" not in used:
            match = re.search(r"(\{.*\}|\[.*\])", goal)
            payload = match.group(1) if match else "{}"
            return "json_validate", {"payload": payload}, "Need to validate JSON."

        if ("time" in lowered or "date" in lowered) and "current_time" not in used:
            return "current_time", {}, "Need the current time."

        if "rag" in lowered or "embedding" in lowered or "agent" in lowered:
            if "lookup_topic" not in used:
                topic = next(
                    (t for t in ("rag", "embedding", "agent") if t in lowered), "agent"
                )
                return "lookup_topic", {"topic": topic}, "Need a definition."

        return None


class Agent:
    """Runs a goal through a policy, enforcing step and tool limits."""

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        policy: Policy | None = None,
        allowed_tools: list[str] | None = None,
        memory: Any | None = None,
    ) -> None:
        self.registry = registry or create_default_registry()
        self.policy = policy or RuleBasedPolicy()
        self.memory = memory
        if allowed_tools is not None:
            missing = set(allowed_tools) - set(self.registry.names())
            if missing:
                raise ValueError(f"unknown allowed tools: {sorted(missing)}")
        self.allowed_tools = allowed_tools

    def run(self, goal: str, max_steps: int = 5) -> Trace:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        trace = Trace(goal=goal)
        if self.memory is not None:
            self.memory.add("user", goal)

        for _ in range(max_steps):
            decision = self.policy.decide(goal, self.registry, trace.steps)
            if decision is None:
                break
            tool_name, arguments, thought = decision
            if self.allowed_tools is not None and tool_name not in self.allowed_tools:
                break
            observation = self.registry.run(tool_name, **arguments)
            trace.steps.append(
                Step(thought, tool_name, arguments, str(observation))
            )
            if self.memory is not None:
                self.memory.add("assistant", {"tool": tool_name, "arguments": arguments})
                self.memory.add("user", str(observation))

        trace.final_answer = (
            trace.steps[-1].observation
            if trace.steps
            else "I don't have a tool that can help with this goal."
        )
        if self.memory is not None:
            self.memory.add("assistant", trace.final_answer)
        return trace
