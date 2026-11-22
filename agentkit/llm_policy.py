"""LLM-backed planner that emits tool calls from OpenAI-compatible APIs."""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Callable

from .agent import Step
from .tools import ToolRegistry

Client = Callable[[list[dict[str, Any]], list[dict[str, Any]]], dict[str, Any]]


def openai_client(api_key: str | None = None, model: str = "gpt-4o-mini") -> Client:
    """Build a chat-completions client with JSON tool-call output."""
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY is required")

    def call(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        payload = json.dumps(
            {
                "model": model,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)

    return call


class LLMPolicy:
    """Planner that asks an LLM which tool to call next.

    The HTTP client is injectable, so the decision logic is fully testable
    without network access or API keys.
    """

    SYSTEM = (
        "You are an agent. Choose one tool to advance the goal. "
        "If the goal is already satisfied, respond with no tool call."
    )

    def __init__(self, registry: ToolRegistry, client: Client | None = None) -> None:
        self.registry = registry
        self.client = client or openai_client()

    def decide(
        self, goal: str, registry: ToolRegistry, steps: list[Step]
    ) -> tuple[str, dict[str, Any], str] | None:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.SYSTEM},
            {"role": "user", "content": goal},
        ]
        for step in steps:
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps({"tool": step.tool, "arguments": step.arguments}),
                }
            )
            messages.append(
                {"role": "user", "content": f"Observation: {step.observation}"}
            )

        response = self.client(messages, registry.schemas())
        choice = response["choices"][0]["message"]
        tool_calls = choice.get("tool_calls") or []
        if not tool_calls:
            return None
        call = tool_calls[0]["function"]
        name = call["name"]
        if name not in registry.names():
            raise ValueError(f"model requested unknown tool: {name}")
        arguments = json.loads(call.get("arguments") or "{}")
        return name, arguments, choice.get("content") or f"Use {name}"
