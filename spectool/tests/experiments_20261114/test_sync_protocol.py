"""CPU check of the step-synchronous loop on the current code, scripted Draft/Main."""
import os, sys, time, threading
os.environ.update(SPECTOOL_IMAGE_FIRST="assess", SPECTOOL_JUDGE_ANCHOR="fast", SPECTOOL_QJ_ONLY="1", SPECTOOL_FAST_ANSWER_HANDOFF_GRACE_SECONDS="1")
sys.path.insert(0, sys.argv[1])
from PIL import Image
from spectool_agent.sync_testing.synchronous_agent import SynchronousSpeculativeAgentLoop as Loop
import spectool_agent.speculative_agent as SA
from spectool_agent.types import AgentConfig, ToolCall
Q = "Which box is red?\nA. left\nB. right\nAnswer with the option letter."
LOG = []
class Draft:
    def __init__(self, script): self.s, self.i = list(script), 0; self.last_truncated = False; self.last_tool_calls = []; self.last_tool_call_ids = []; self.last_assistant_msg = None
    def generate_full(self, msgs, **kw):
        LOG.append(("draft_start", time.time())); time.sleep(0.2); st = self.s[min(self.i, len(self.s) - 1)]; self.i += 1
        if st == "tool2": self.last_tool_calls = [ToolCall("sandbox", {"code": "a"}), ToolCall("sandbox", {"code": "b"})]; self.last_tool_call_ids = ["c1", "c2"]; t = "tools"
        elif st == "tool": self.last_tool_calls = [ToolCall("sandbox", {"code": "x"})]; self.last_tool_call_ids = ["c"]; t = "tool"
        else: self.last_tool_calls, self.last_tool_call_ids = [], []; t = f"<answer>{st}</answer>"
        self.last_assistant_msg = {"role": "assistant", "content": t, "tool_calls": [{"id": c, "type": "function", "function": {"name": "sandbox", "arguments": "{}"}} for c in self.last_tool_call_ids]}
        return t
class Tools:
    def __init__(self): self.n = 0
    def set_input_images(self, i): pass
    def dispatch(self, tc, image): self.n += 1; return "crop", []
    def cleanup(self): pass
class Main:
    def __init__(self, fast_ok, qj, ver): self.fast_ok, self.qj, self.ver = fast_ok, list(qj), ver
    def fast_answer_gate(self, messages, choices=None, cancel_token=None):
        LOG.append(("fast_start", time.time())); time.sleep(0.3); LOG.append(("fast_end", time.time()))
        if self.fast_ok: return {"answerable": True, "answer": "B", "answerability_score": 0.99, "answerability_reason": "answer", "raw": "", "verification_score": 0.99}
        return {"answerable": False, "answer": None, "answerability_score": None, "answerability_reason": "insufficient", "raw": ""}
    def judge_good_bad(self, messages, cancel_token=None, **kw):
        last = messages[-1]["content"][0]["text"]
        if "CANDIDATE" in last or "VERIFICATION" in last: LOG.append(("verifier", time.time())); s = self.ver
        else: LOG.append(("judge_start", time.time())); time.sleep(0.4); LOG.append(("judge_end", time.time())); s = self.qj.pop(0) if self.qj else 0.9
        return {"score": s, "label": "GOOD" if s >= .5 else "BAD", "raw": "", "log_good": -.1, "log_bad": -3}
def run(script, m, fast=True):
    LOG.clear(); os.environ["SPECTOOL_FAST_ANSWER_ENABLED"] = "1" if fast else "0"
    cfg = AgentConfig(api_base="x", model_name="m", quality_judge_enabled=True, quality_judge_threshold=0.7, answerability_threshold=0.9, draft_max_steps=8, max_steps=8)
    tools = Tools(); ag = Loop(m, Draft(script), tools, cfg); fb = []
    ag._run_fallback_agent = lambda *a, **k: (fb.append(k.get("trigger")), setattr(a[3], "final_answer", "FB"))
    try: t = ag.run(Image.new("RGB", (32, 32)), Q, sample_id="t")
    finally: ag.shutdown()
    return t, fb, tools, list(LOG)
ok = True
def check(n, c, d=""):
    global ok; ok &= c; print("PASS" if c else "FAIL", n, d)
t, fb, tools, L = run(["tool", "A"], Main(True, [], 0.9))
check("fast accepted before any Draft request", t.final_answer == "B" and t.stop_reason == "fast_main_answer" and not any(e == "draft_start" for e, _ in L), f"{t.final_answer} {[e for e,_ in L]}")
t, fb, tools, L = run(["tool", "tool", "A"], Main(False, [0.9, 0.9], 0.9))
ev = [e for e, _ in L]
first_draft = min(x for e, x in L if e == "draft_start"); fast_end = [x for e, x in L if e == "fast_end"][0]
# second Draft request must start only after the first judge finished
ds = [x for e, x in L if e == "draft_start"]; je = [x for e, x in L if e == "judge_end"]
check("fast rejected: Draft starts after fast; each Draft step waits for the judge", first_draft >= fast_end and len(ds) >= 2 and ds[1] >= je[0] and t.final_answer == "A", f"{t.final_answer} {ev}")
check("Draft answer goes through the verifier", "verifier" in ev and t.stop_reason == "main_answer_draft_evidence_verified", t.stop_reason)
t, fb, tools, L = run(["tool", "tool", "tool", "A"], Main(False, [0.1, 0.1], 0.9))
check("two BAD steps -> fallback, Draft stopped", fb == ["quality_bad_streak"] and sum(e == "draft_start" for e, _ in L) == 2, f"{fb} {[e for e,_ in L]}")
t, fb, tools, L = run(["tool2", "A"], Main(False, [0.9], 0.9))
check("only the first of parallel tool actions is executed", tools.n == 1 and t.final_answer == "A", f"tools run {tools.n}")
t, fb, tools, L = run(["tool", "A"], Main(True, [0.9], 0.9), fast=False)
check("fast off: no fast request, Draft + judge + verifier", not any(e == "fast_start" for e, _ in L) and t.final_answer == "A", f"{[e for e,_ in L]}")
print("ALL PASSED" if ok else "FAILED")
