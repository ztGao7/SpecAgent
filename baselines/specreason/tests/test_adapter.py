import copy
import json
import unittest
import sys
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import Mock

runtime = Path(__file__).resolve().parents[1] / "runtime"
if runtime.exists():
    sys.path[:0] = [str(runtime), str(runtime / "VLMEvalKit")]

from specreason_adapter.core import (BudgetExceeded, Config, InfrastructureError, Ledger,
    ProtocolError, SpecReasonAgent, Turn)
from specreason_adapter.model import Endpoint, parse_score, parse_turn
from specreason_adapter.prompts import GENERATION_PROMPT, SCORE_ASSISTANT_PREFIX, scoring_messages
from spectool_agent.prompts import SYSTEM_PROMPT
from spectool_agent.types import ToolCall


def answer(value="A"):
    return Turn({"role": "assistant", "content": f"<answer>{value}</answer>"}, answer=value)


def action(code, cid="c1"):
    call = ToolCall("sandbox", {"code": code})
    return Turn({"role": "assistant", "content": "inspect", "tool_calls": [{
        "id": cid, "type": "function", "function": {"name": "sandbox", "arguments": json.dumps(call.params)}}]}, [call], [cid])


class FakeEndpoint:
    def __init__(self, role, turns=(), scores=(), tokens=2):
        self.role, self.turns, self.scores, self.tokens = role, iter(turns), iter(scores), tokens
        self.histories, self.scored = [], []

    def generate(self, history, schemas, ledger, phase):
        limit = ledger.allowance(self.tokens)
        self.histories.append(copy.deepcopy(history))
        ledger.record_request({"role": self.role, "phase": phase, "usage": {
            "completion_tokens": limit, "prompt_tokens": 10}})
        result = next(self.turns)
        if isinstance(result, Exception):
            raise result
        return result

    def score(self, history, candidate, schemas, ledger):
        ledger.allowance(1)
        self.scored.append(copy.deepcopy(candidate.message))
        ledger.record_request({"role": self.role, "phase": "score", "usage": {
            "completion_tokens": 1, "prompt_tokens": 20}})
        return next(self.scores)


class FakeTools:
    def __init__(self):
        self.state, self.executed, self.initialized = {}, [], 0

    def set_input_images(self, images):
        self.initialized += 1

    def validate_call(self, call):
        return None if call.tool_name == "sandbox" and call.params.get("code") else "bad_call"

    def dispatch(self, call, image):
        self.executed.append(call.params["code"])
        exec(call.params["code"], {}, self.state)
        return str(self.state), []


HISTORY = [{"role": "system", "content": GENERATION_PROMPT},
    {"role": "user", "content": [{"type": "text", "text": "Which?\nA. yes\nB. no"}]}]


class CoreTests(unittest.TestCase):
    def run_agent(self, draft_turns, scores, rewrites=(), config=None):
        draft = FakeEndpoint("draft", draft_turns)
        main = FakeEndpoint("main", rewrites, scores)
        tools = FakeTools()
        agent = SpecReasonAgent(main, draft, tools, [], config or Config(),
            lambda cid, obs, imgs: [{"role": "tool", "tool_call_id": cid, "content": obs}])
        result = agent.run(HISTORY, [object()], "sample-1")
        return result, tools, main, draft

    def test_accept_executes_exactly_once_then_returns_to_draft(self):
        r, tools, main, draft = self.run_agent([action("x=3"), answer()], [7, 9])
        self.assertEqual(tools.executed, ["x=3"])
        self.assertEqual(r["final_answer"], "A")
        self.assertEqual(len(draft.histories), 2)
        self.assertEqual(len(main.histories), 0)
        self.assertEqual(len(main.scored), 2)  # Final candidate is also judged.

    def test_rejected_action_never_executes_or_enters_rewrite_context(self):
        r, tools, main, draft = self.run_agent([action("forbidden=1"), answer()],
            [2, 8], [action("accepted=2")])
        self.assertEqual(tools.executed, ["accepted=2"])
        self.assertNotIn("forbidden", json.dumps(main.histories))
        self.assertNotIn("forbidden", json.dumps(draft.histories[1]))
        self.assertEqual(r["accounting"]["turns_committed"], 2)

    def test_rewrite_reuses_accepted_tool_state(self):
        r, tools, main, draft = self.run_agent(
            [action("x=3"), action("x=999", "c2"), answer()], [8, 1, 8],
            [action("y=x+2", "rewrite")])
        self.assertEqual(tools.state, {"x": 3, "y": 5})
        self.assertEqual(tools.initialized, 1)
        self.assertIn("x", main.histories[0][-1]["content"])
        self.assertEqual(r["accounting"]["output_tokens"], 11)

    def test_invalid_score_rewrites_step(self):
        r, _, main, _ = self.run_agent([answer("B")], [None], [answer("A")])
        self.assertEqual(r["final_answer"], "A")
        self.assertFalse(r["steps"][0]["accepted_draft"])

    def test_bare_text_feedback_matches_baseline_and_consumes_a_turn(self):
        from spectool_agent.agent import _diagnose_tool_call
        bare = Turn({'role': 'assistant', 'content': 'The correct option is A.'})
        main = FakeEndpoint('main', scores=[9, 9])
        draft = FakeEndpoint('draft', [bare, answer()])
        agent = SpecReasonAgent(main, draft, FakeTools(), [], Config(max_turns=2),
            lambda *args: [], format_feedback=lambda turn: _diagnose_tool_call(turn.message['content']))
        result = agent.run(HISTORY, [object()])
        self.assertEqual(result['final_answer'], 'A')
        self.assertEqual(result['accounting']['turns_committed'], 2)
        self.assertEqual(result['accounting']['output_tokens'], 6)
        self.assertEqual(draft.histories[1][-1]['role'], 'user')
        self.assertEqual(draft.histories[1][-1]['content'][0]['text'], _diagnose_tool_call(bare.message['content']))
        self.assertEqual(result['steps'][0]['protocol_feedback'], _diagnose_tool_call(bare.message['content']))

    def test_max_turns_is_shared_no_new_budget_on_rejection(self):
        r, tools, _, _ = self.run_agent([action("bad=1")], [0], [action("x=1")], Config(max_turns=1))
        self.assertEqual(r["stop_reason"], "max_turns")
        self.assertEqual(tools.executed, ["x=1"])

    def test_all_spent_tokens_include_rejected_candidate_and_judge(self):
        r, _, _, _ = self.run_agent([answer("B")], [0], [answer("A")])
        self.assertEqual(r["accounting"]["output_tokens"], 5)
        self.assertEqual(r["accounting"]["roles"]["main"]["completion_tokens"], 3)

    def test_token_budget_can_stop_before_rewrite(self):
        r, _, main, _ = self.run_agent([answer("B")], [0], [], Config(max_total_output_tokens=3))
        self.assertEqual(r["stop_reason"], "output_token_budget")
        self.assertEqual(len(main.histories), 0)
        self.assertIsNone(r["final_answer"])

    def test_multi_call_tool_budget_is_checked_before_any_execution(self):
        candidate = action("x=1")
        candidate.calls += [ToolCall("sandbox", {"code": "y=2"})]
        candidate.call_ids += ["c2"]
        r, tools, _, _ = self.run_agent([candidate], [9], config=Config(max_tool_calls=1))
        self.assertEqual(tools.executed, [])
        self.assertEqual(r["stop_reason"], "tool_call_budget")

    def test_invalid_tool_cannot_run_even_with_high_score(self):
        invalid = action("")
        r, tools, _, _ = self.run_agent([invalid], [9], [answer()])
        self.assertEqual(tools.executed, [])
        self.assertEqual(r["steps"][0]["source"], "main_rewrite")

    def test_bad_main_rewrite_stops_without_free_retries(self):
        bad = Turn({"role": "assistant", "content": ""}, valid=False, error="truncated_turn")
        r, _, main, _ = self.run_agent([answer()], [0], [bad])
        self.assertEqual(r["stop_reason"], "protocol_error")
        self.assertEqual(len(main.histories), 1)

    def test_main_only_does_not_query_draft_or_judge(self):
        r, _, main, draft = self.run_agent([], [], [answer()], Config(mode="main-only"))
        self.assertEqual(r["final_answer"], "A")
        self.assertEqual(draft.histories, [])
        self.assertEqual(main.scored, [])

    def test_fatal_service_failure_propagates_with_partial_record(self):
        main = FakeEndpoint("main")
        draft = FakeEndpoint("draft", [InfrastructureError("down")])
        agent = SpecReasonAgent(main, draft, FakeTools(), [], Config(), lambda *a: [])
        with self.assertRaises(InfrastructureError):
            agent.run(HISTORY, [object()])
        self.assertEqual(agent.last_result["stop_reason"], "infrastructure_error")
        self.assertIn("accounting", agent.last_result)

    def test_usage_missing_is_not_silently_zero(self):
        ledger = Ledger(Config())
        with self.assertRaises(ProtocolError):
            ledger.record_request({"role": "draft", "usage": None})
        self.assertFalse(ledger.usage_complete)

    def test_generation_prompt_is_shared_neutral_prompt(self):
        self.assertEqual(GENERATION_PROMPT, SYSTEM_PROMPT)
        self.assertNotIn("[Spectool draft role]", GENERATION_PROMPT)

    def test_score_message_does_not_create_pending_native_tool_call(self):
        messages = scoring_messages(HISTORY, action("x=1").message)
        self.assertEqual(messages[:-2], HISTORY)
        self.assertEqual(messages[-2]["role"], "user")
        self.assertEqual(messages[-1], {"role": "assistant", "content": SCORE_ASSISTANT_PREFIX})
        self.assertTrue(all("tool_calls" not in message for message in messages[-2:]))


def choice(content, reasoning="", calls=None, finish="stop"):
    raw = {"content": content, "reasoning_content": reasoning, "tool_calls": calls or []}
    return NS(message=NS(content=content, model_dump=lambda: raw), finish_reason=finish)


class ParsingAndTransportTests(unittest.TestCase):
    def test_clipped_deadline_timeout_is_budget_exhaustion_not_service_failure(self):
        ledger = Ledger(Config(wall_seconds=10))
        def time_out(**kwargs):
            ledger.started -= 11
            raise TimeoutError("simulated sample deadline")
        client = Mock()
        client.chat.completions.create.side_effect = time_out
        endpoint = Endpoint(role="main", model="test", base_url="http://unused/v1",
            timeout=180, thinking=False, client=client)
        with self.assertRaises(BudgetExceeded):
            endpoint.generate(HISTORY, [], ledger, "main_rewrite")
        self.assertFalse(ledger.usage_complete)
        self.assertEqual(len(ledger.requests), 1)

    def test_score_is_one_ascii_digit(self):
        for bad in (None, "", "10", "7/9", "Score: 7", "７", "²"):
            self.assertIsNone(parse_score(bad))
        self.assertEqual(parse_score(" 7\n"), 7)

    def test_reasoning_is_preserved_but_not_used_as_final_answer(self):
        turn = parse_turn(choice("<answer>B</answer>", "<answer>A</answer>"), "A. yes\nB. no")
        self.assertEqual(turn.answer, "B")
        self.assertEqual(turn.message["reasoning_content"], "<answer>A</answer>")

    def test_truncated_candidate_is_not_a_complete_turn(self):
        self.assertFalse(parse_turn(choice("<answer>A</answer>", finish="length"), "").valid)

    def test_native_call_arguments_are_not_silently_replaced_with_empty_dict(self):
        calls = [{"id": "c", "function": {"name": "sandbox", "arguments": "{broken"}}]
        turn = parse_turn(choice("", calls=calls), "")
        self.assertFalse(turn.valid)
        self.assertIn("malformed_tool_call", turn.error)

    def test_score_request_pins_no_thinking_and_disables_tools(self):
        response = Mock()
        response.usage.model_dump.return_value = {"completion_tokens": 1, "prompt_tokens": 12}
        response.choices = [choice("7", finish="length")]
        response.model_dump.return_value = {"choices": [], "usage": {"completion_tokens": 1}}
        client = Mock()
        client.chat.completions.create.return_value = response
        endpoint = Endpoint(role="main", model="test", base_url="http://unused/v1",
            timeout=180, thinking=True, client=client)
        ledger = Ledger(Config())
        score = endpoint.score(HISTORY, action("x=1"), [{"type": "function"}], ledger)
        req = client.chat.completions.create.call_args.kwargs
        self.assertEqual(score, 7)
        self.assertEqual(req["max_tokens"], 1)
        self.assertEqual(req["tool_choice"], "none")
        self.assertFalse(req["extra_body"]["chat_template_kwargs"]["enable_thinking"])
        self.assertFalse(req["extra_body"]["add_generation_prompt"])
        self.assertTrue(req["extra_body"]["continue_final_message"])
        self.assertEqual(req["extra_body"]["structured_outputs"], {"regex": "[0-9]"})
        self.assertEqual(req["messages"][-1]["content"], SCORE_ASSISTANT_PREFIX)
        self.assertEqual(ledger.output_tokens, 1)


if __name__ == "__main__":
    unittest.main()
