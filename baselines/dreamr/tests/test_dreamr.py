"""DREAM-R mode: TBVM threshold, parallel rewrite, isolation of rejected candidates."""
import sys, threading, time, types, unittest
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "spectool"))
from dreamr_adapter.core import Config, Ledger, ProtocolError, DreamRAgent, Turn
from dreamr_adapter.model import Endpoint, PendingTurn, positive_probability
from dreamr_adapter.prompts import verification_messages


@dataclass
class Call:
    code: str
    tool_name: str = "python"
    params: dict = None
    def __post_init__(self): self.params = {"code": self.code}


def action(code):
    return Turn({"role": "assistant", "content": "", "tool_calls": [{"id": "c" + code}]}, [Call(code)], ["c" + code])


def answer(text="A"):
    return Turn({"role": "assistant", "content": f"<answer>{text}</answer>"}, answer=text)


class FakePending:
    def __init__(self, owner, turn): self.owner, self.turn, self.state = owner, turn, "running"
    def cancel(self): self.state = "cancelled"; self.owner.cancelled += 1; return "cancelled"
    def result(self):
        self.owner.used += 1
        if isinstance(self.turn, Exception): raise self.turn
        return self.turn


class FakeMain:
    def __init__(self, probs, rewrites, warmups=()):
        self.probs, self.rewrites, self.warmups = list(probs), list(rewrites), list(warmups)
        self.cancelled = self.used = self.started = 0
        self.verify_histories, self.rewrite_histories = [], []
    def verify(self, history, candidate, schemas, ledger):
        self.verify_histories.append(list(history)); return self.probs.pop(0)
    def start_generate(self, history, schemas, ledger, phase):
        self.started += 1; self.rewrite_histories.append(list(history))
        return FakePending(self, self.rewrites.pop(0) if self.rewrites else answer("unused"))
    def generate(self, history, schemas, ledger, phase): return self.warmups.pop(0)


class FakeDraft:
    def __init__(self, turns): self.turns, self.calls = list(turns), 0
    def generate(self, history, schemas, ledger, phase): self.calls += 1; return self.turns.pop(0)


class FakeTools:
    def __init__(self): self.executed = []
    def set_input_images(self, images): pass
    def validate_call(self, call): return ""
    def dispatch(self, call, image): self.executed.append(call.params["code"]); return "ok", []


def run(drafts, probs, rewrites=(), config=None, warmups=()):
    main, draft, tools = FakeMain(probs, rewrites, warmups), FakeDraft(drafts), FakeTools()
    agent = DreamRAgent(main, draft, tools, [], config or Config(mode="dreamr"),
        lambda cid, obs, imgs: [{"role": "tool", "tool_call_id": cid, "content": obs}])
    return agent.run([{"role": "user", "content": "q"}], [object()], "s"), tools, main, draft


class DreamRFlow(unittest.TestCase):
    def test_accept_at_threshold_cancels_rewrite_and_executes_draft(self):
        r, tools, main, _ = run([action("a"), answer("B")], [0.7, 0.95])
        self.assertEqual((r["final_answer"], tools.executed), ("B", ["a"]))
        self.assertEqual((main.started, main.cancelled, main.used), (2, 2, 0))
        self.assertEqual([s["source"] for s in r["steps"]], ["draft", "draft"])

    def test_below_threshold_uses_parallel_rewrite_and_never_executes_draft(self):
        r, tools, main, _ = run([action("bad"), answer("B")], [0.69, 0.9], [action("good")])
        self.assertEqual(tools.executed, ["good"])
        self.assertEqual(r["steps"][0]["source"], "main_rewrite")
        self.assertFalse(any("bad" in str(m) for h in main.rewrite_histories + main.verify_histories for m in h))

    def test_rewrite_starts_before_verification_finishes(self):
        order = []
        class Main(FakeMain):
            def start_generate(self, *a): order.append("rewrite_started"); return super().start_generate(*a)
            def verify(self, *a): order.append("verify"); return super().verify(*a)
        main, draft, tools = Main([0.1], [answer("C")]), FakeDraft([answer("A")]), FakeTools()
        r = DreamRAgent(main, draft, tools, [], Config(mode="dreamr"), lambda *a: []).run(
            [{"role": "user", "content": "q"}], [object()])
        self.assertEqual((order, r["final_answer"]), (["rewrite_started", "verify"], "C"))

    def test_missing_probability_rejects(self):
        r, _, main, _ = run([answer("A")], [None], [answer("C")])
        self.assertEqual((r["final_answer"], main.used), ("C", 1))

    def test_invalid_candidate_is_rejected_even_with_high_probability(self):
        bad = Turn({"role": "assistant", "content": "x"}, valid=False, error="truncated_turn")
        r, _, _, _ = run([bad], [0.99], [answer("C")])
        self.assertEqual((r["final_answer"], r["steps"][0]["candidate_error"]), ("C", "truncated_turn"))

    def test_verify_failure_cancels_rewrite(self):
        class Main(FakeMain):
            def verify(self, *a): raise ProtocolError("boom")
        main = Main([], [answer("C")])
        r = DreamRAgent(main, FakeDraft([answer("A")]), FakeTools(), [], Config(mode="dreamr"),
            lambda *a: []).run([{"role": "user", "content": "q"}], [object()])
        self.assertEqual((r["stop_reason"], main.cancelled), ("protocol_error", 1))

    def test_first_main_turns_skip_draft(self):
        r, tools, main, draft = run([answer("B")], [0.9], [], Config(mode="dreamr", first_main_turns=1),
            warmups=[action("w")])
        self.assertEqual((tools.executed, draft.calls, r["steps"][0]["source"]), (["w"], 1, "main_warmup"))

    def test_threshold_bounds(self):
        with self.assertRaises(ValueError): Config(mode="dreamr", accept_threshold=1.5)


def lp(token, logprob): return types.SimpleNamespace(token=token, logprob=logprob, top_logprobs=None)


def choice_with(tops):
    first = types.SimpleNamespace(token=tops[0][0], logprob=tops[0][1], top_logprobs=[lp(*t) for t in tops])
    return types.SimpleNamespace(logprobs=types.SimpleNamespace(content=[first]), finish_reason="length",
        message=types.SimpleNamespace(content=tops[0][0]))


class Accounting(unittest.TestCase):
    def test_aborted_rewrite_does_not_claim_complete_token_usage(self):
        ledger = Ledger(Config(mode="dreamr"))
        ledger.record_cancelled({"role": "main", "phase": "main_rewrite", "status": "cancelled"})
        self.assertFalse(ledger.summary()["usage_complete"])
        self.assertEqual(ledger.summary()["output_tokens"], 0)


class Probability(unittest.TestCase):
    def test_renormalised_over_two_labels(self):
        import math
        p = positive_probability(choice_with([("positive", math.log(0.6)), ("negative", math.log(0.2)), ("The", math.log(0.2))]))
        self.assertAlmostEqual(p, 0.75)

    def test_subword_prefixes_and_case(self):
        import math
        p = positive_probability(choice_with([(" Pos", math.log(0.3)), ("neg", math.log(0.1)), ("n", math.log(0.2))]))
        self.assertAlmostEqual(p, 0.5)

    def test_no_label_or_no_logprobs_is_none(self):
        self.assertIsNone(positive_probability(choice_with([("The", -0.1)])))
        self.assertIsNone(positive_probability(types.SimpleNamespace(logprobs=None)))

    def test_nan_is_ignored(self):
        self.assertIsNone(positive_probability(choice_with([("positive", float("nan"))])))


class FakeClient:
    """Sync client double; records requests and can block to emulate a slow rewrite."""
    def __init__(self, content, release=None):
        self.content, self.release, self.requests = content, release, []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))
    def create(self, **kwargs):
        self.requests.append(kwargs)
        if self.release is not None: self.release.wait(5)
        msg = types.SimpleNamespace(content=self.content, model_dump=lambda: {"content": self.content})
        ch = types.SimpleNamespace(message=msg, finish_reason="stop", logprobs=None)
        return types.SimpleNamespace(choices=[ch], usage=types.SimpleNamespace(
            model_dump=lambda: {"completion_tokens": 3, "prompt_tokens": 5}),
            model_dump=lambda: {"id": "x"})


class Transport(unittest.TestCase):
    def test_verify_request_is_one_constrained_token_without_tools_or_thinking(self):
        client = FakeClient("positive")
        ep = Endpoint(role="main", model="m", base_url="http://unused/v1", timeout=5, thinking=True, client=client)
        ep.verify([{"role": "user", "content": "q"}], answer("A"), [{"type": "function"}], Ledger(Config(mode="dreamr")))
        k = client.requests[0]
        self.assertEqual((k["max_tokens"], k["tool_choice"], k["top_logprobs"]), (1, "none", 20))
        self.assertFalse(k["extra_body"]["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(k["extra_body"]["structured_outputs"], {"choice": ["positive", "negative"]})
        self.assertEqual(k["messages"][-1], {"role": "assistant", "content": "The answer is: "})
        self.assertNotIn("repetition_penalty", k["extra_body"])

    def test_verification_message_serialises_candidate_as_data(self):
        cand = {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]}
        msgs = verification_messages([{"role": "user", "content": "q"}], cand)
        self.assertEqual([m["role"] for m in msgs], ["user", "user", "assistant"])
        self.assertIn('"tool_calls"', msgs[1]["content"])

    def test_pending_turn_result_and_accounting(self):
        client = FakeClient("<answer>B</answer>")
        ep = Endpoint(role="main", model="m", base_url="http://unused/v1", timeout=5, thinking=False, client=client)
        ledger = Ledger(Config(mode="dreamr"))
        pending = ep.start_generate([{"role": "user", "content": [{"type": "text", "text": "q"}]}], [], ledger, "main_rewrite")
        self.assertIsInstance(pending, PendingTurn)
        self.assertEqual(pending.result().message["content"], "<answer>B</answer>")
        self.assertEqual((ledger.output_tokens, ledger.requests[0]["phase"]), (3, "main_rewrite"))

    def test_start_generate_returns_before_the_request_finishes(self):
        release = threading.Event()
        client = FakeClient("<answer>B</answer>", release)
        ep = Endpoint(role="main", model="m", base_url="http://unused/v1", timeout=5, thinking=False, client=client)
        t0 = time.perf_counter()
        pending = ep.start_generate([{"role": "user", "content": "q"}], [], Ledger(Config(mode="dreamr")), "main_rewrite")
        self.assertLess(time.perf_counter() - t0, 1.0)
        release.set()
        self.assertEqual(pending.cancel(), "completed_discarded")


if __name__ == "__main__":
    unittest.main()
