import unittest

import json

from agentkit import (
    Agent,
    ConversationMemory,
    LLMPolicy,
    RuleBasedPolicy,
    Tool,
    ToolRegistry,
    __version__,
    calculator,
    create_default_registry,
    current_time,
    json_validate,
    lookup_topic,
    text_stats,
)


class ToolTests(unittest.TestCase):
    def test_registry(self):
        registry = create_default_registry()
        self.assertIn("calculator", registry.names())
        self.assertEqual(len(registry.schemas()), 5)
        self.assertEqual(registry.schemas()[0]["type"], "function")

    def test_duplicate_registration(self):
        registry = ToolRegistry()
        tool = Tool("noop", "noop", {"type": "object"}, lambda: None)
        registry.register(tool)
        with self.assertRaises(ValueError):
            registry.register(tool)
        with self.assertRaises(KeyError):
            registry.get("missing")

    def test_safe_calculator(self):
        self.assertEqual(calculator("2 + 3 * 4"), 14)
        self.assertEqual(calculator("(2 + 3) * 4"), 20)
        self.assertEqual(calculator("-5 + 2"), -3)
        with self.assertRaises(ValueError):
            calculator("__import__('os')")
        with self.assertRaises(ValueError):
            calculator("2 ** 1000")

    def test_builtin_tools(self):
        stats = text_stats("Hello world. Another one!")
        self.assertEqual(stats["words"], 4)
        self.assertEqual(stats["sentences"], 2)
        self.assertTrue(json_validate('{"a": 1}'))
        with self.assertRaises(ValueError):
            json_validate("{bad")
        self.assertIn("grounds generation", lookup_topic("RAG"))
        self.assertGreater(len(current_time()), 10)


class AgentTests(unittest.TestCase):
    def test_agent_solves_math_goal(self):
        trace = Agent().run("Calculate 12 * 8 + 4")
        self.assertEqual(trace.final_answer, "100")
        self.assertEqual(len(trace.steps), 1)
        self.assertEqual(trace.steps[0].tool, "calculator")
        self.assertIn("Answer: 100", trace.as_lines()[-1])

    def test_agent_uses_multiple_tools(self):
        trace = Agent().run("Calculate 6 * 7 and explain RAG")
        self.assertEqual([s.tool for s in trace.steps], ["calculator", "lookup_topic"])
        self.assertIn("grounds generation", trace.final_answer)

    def test_allowed_tools_limits_actions(self):
        trace = Agent(allowed_tools=["current_time"]).run("Calculate 2 + 2")
        self.assertEqual(trace.steps, [])

    def test_max_steps(self):
        with self.assertRaises(ValueError):
            Agent().run("Calculate 1 + 1", max_steps=0)
        trace = Agent().run("Calculate 1 + 1", max_steps=1)
        self.assertEqual(len(trace.steps), 1)

    def test_policy_with_no_decision(self):
        class NoOpPolicy:
            def decide(self, goal, registry, steps):
                return None

        trace = Agent(policy=NoOpPolicy()).run("anything")
        self.assertEqual(trace.steps, [])
        self.assertEqual(trace.final_answer, "I don't have a tool that can help with this goal.")

    def test_unknown_allowed_tool(self):
        with self.assertRaises(ValueError):
            Agent(allowed_tools=["nope"])


class V02Tests(unittest.TestCase):
    def test_version(self):
        self.assertEqual(__version__, "0.2.0")

    def test_memory_ring_buffer(self):
        memory = ConversationMemory(capacity=3)
        for i in range(5):
            memory.add("user", i)
        self.assertEqual(len(memory), 3)
        self.assertEqual([i.content for i in memory.recent()], [2, 3, 4])
        self.assertEqual(len(memory.as_messages()), 3)
        memory.clear()
        self.assertEqual(len(memory), 0)

    def test_agent_writes_to_memory(self):
        memory = ConversationMemory()
        trace = Agent(memory=memory).run("Calculate 2 + 3")
        self.assertEqual(trace.final_answer, "5")
        self.assertEqual(memory.recent()[0].content, "Calculate 2 + 3")
        self.assertEqual(memory.recent()[-1].content, "5")

    def test_llm_policy_with_fake_client(self):
        registry = create_default_registry()
        seen = {}

        def fake_client(messages, tools):
            seen["messages"] = messages
            seen["tools"] = tools
            return {
                "choices": [
                    {
                        "message": {
                            "content": "compute it",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "calculator",
                                        "arguments": json.dumps({"expression": "6*7"}),
                                    }
                                }
                            ],
                        }
                    }
                ]
            }

        policy = LLMPolicy(registry, client=fake_client)
        trace = Agent(registry=registry, policy=policy).run("What is 6*7?")
        self.assertEqual(trace.final_answer, "42")
        self.assertEqual(seen["tools"][0]["type"], "function")
        self.assertEqual(seen["messages"][0]["role"], "system")

    def test_llm_policy_no_tool_call(self):
        registry = create_default_registry()
        policy = LLMPolicy(
            registry,
            client=lambda m, t: {"choices": [{"message": {"content": "done"}}]},
        )
        trace = Agent(registry=registry, policy=policy).run("hello")
        self.assertEqual(trace.steps, [])

    def test_llm_policy_rejects_unknown_tool(self):
        registry = create_default_registry()
        policy = LLMPolicy(
            registry,
            client=lambda m, t: {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"function": {"name": "nope", "arguments": "{}"}}
                            ]
                        }
                    }
                ]
            },
        )
        with self.assertRaises(ValueError):
            Agent(registry=registry, policy=policy).run("anything")


if __name__ == "__main__":
    unittest.main()
