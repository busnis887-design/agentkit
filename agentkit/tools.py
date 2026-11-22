"""Tool definitions, registry, and safe built-in tools."""

from __future__ import annotations

import ast
import json
import operator
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]

    def run(self, **kwargs: Any) -> Any:
        return self.func(**kwargs)

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"unknown tool: {name}") from exc

    def run(self, name: str, **kwargs: Any) -> Any:
        return self.get(name).run(**kwargs)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.openai_schema() for tool in self._tools.values()]


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


def calculator(expression: str) -> float:
    """Evaluate arithmetic without ``eval``: numbers and + - * / % ** only."""
    tree = ast.parse(expression, mode="eval")
    result = _safe_eval(tree.body)
    return round(result, 10)


def text_stats(text: str) -> dict[str, int]:
    return {
        "characters": len(text),
        "words": len(re.findall(r"\b\w+\b", text)),
        "sentences": len([s for s in re.split(r"[.!?]+", text) if s.strip()]),
    }


def json_validate(payload: str) -> bool:
    json.loads(payload)
    return True


def current_time() -> str:
    return datetime.now(timezone.utc).isoformat()


_KNOWLEDGE = {
    "rag": "RAG grounds generation in retrieved documents to reduce hallucination.",
    "embedding": "Embeddings map text to vectors for semantic similarity search.",
    "agent": "An agent selects and calls tools in a loop to accomplish a goal.",
}


def lookup_topic(topic: str) -> str:
    return _KNOWLEDGE.get(topic.lower().strip(), f"No local entry for '{topic}'.")


def create_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="calculator",
            description="Evaluate an arithmetic expression.",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
            func=calculator,
        )
    )
    registry.register(
        Tool(
            name="text_stats",
            description="Count characters, words and sentences in a text.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            func=text_stats,
        )
    )
    registry.register(
        Tool(
            name="json_validate",
            description="Check whether a string is valid JSON.",
            parameters={
                "type": "object",
                "properties": {"payload": {"type": "string"}},
                "required": ["payload"],
            },
            func=json_validate,
        )
    )
    registry.register(
        Tool(
            name="current_time",
            description="Return the current UTC time in ISO format.",
            parameters={"type": "object", "properties": {}},
            func=current_time,
        )
    )
    registry.register(
        Tool(
            name="lookup_topic",
            description="Look up a short local AI engineering definition.",
            parameters={
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"],
            },
            func=lookup_topic,
        )
    )
    return registry
