"""agentkit - a small, auditable tool-calling agent runtime."""

from .agent import Agent, RuleBasedPolicy, Step, Trace
from .llm_policy import LLMPolicy, openai_client
from .memory import ConversationMemory, MemoryItem
from .tools import Tool, ToolRegistry, calculator, create_default_registry, current_time, json_validate, lookup_topic, text_stats

__version__ = "0.2.0"

__all__ = [
    "Agent",
    "ConversationMemory",
    "LLMPolicy",
    "MemoryItem",
    "RuleBasedPolicy",
    "Step",
    "Tool",
    "ToolRegistry",
    "Trace",
    "calculator",
    "create_default_registry",
    "current_time",
    "json_validate",
    "lookup_topic",
    "openai_client",
    "text_stats",
]
