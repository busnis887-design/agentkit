"""agentkit - a small, auditable tool-calling agent runtime."""

from .agent import Agent, RuleBasedPolicy, Step, Trace
from .tools import Tool, ToolRegistry, calculator, create_default_registry, current_time, json_validate, lookup_topic, text_stats

__version__ = "0.1.0"

__all__ = [
    "Agent",
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
    "text_stats",
]
