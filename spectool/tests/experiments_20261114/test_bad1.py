"""CPU check of SPECTOOL_MAX_CONSECUTIVE_QUALITY_BAD=1 (first BAD step -> Main fallback), async and sync loops."""
import os, sys, time
os.environ.update(SPECTOOL_FAST_ANSWER_ENABLED="0", SPECTOOL_IMAGE_FIRST="assess", SPECTOOL_JUDGE_ANCHOR="fast",
                  SPECTOOL_QJ_ONLY="1", SPECTOOL_MAX_CONSECUTIVE_QUALITY_BAD="1")
sys.path.insert(0, sys.argv[1])
from PIL import Image
import spectool_agent.speculative_agent as SA
from spectool_agent.sync_testing.synchronous_agent import SynchronousSpeculativeAgentLoop
from spectool_agent.types import AgentConfig, ToolCall
Q = "Which box is red?\nA. left\nB. right\nAnswer with the option letter."
class Draft:
    def __init__(self, script): self.s, self.i, self.n = list(script), 0, 0; self.last_truncated = False; self.last_tool_calls = []; self.last_tool_call_ids = []; self.last_assistant_msg = None
    def generate_full(self, msgs, **kw):
        time.sleep(0.3); st = self.s[min(self.i, len(self.s) - 1)]; self.i += 1; self.n += 1
        if st == "tool": self.last_tool_calls = [ToolCall("sandbox", {"code": "x"})]; self.last_tool_call_ids = ["c"]; t = "tool"
        else: self.last_tool_calls, self.last_tool_call_ids = [], []; t = f"<answer>{st}</answer>"
        self.last_assistant_msg = {"role": "assistant", "content": t, "tool_calls": [{"id": c, "type": "function", "function": {"name": "sandbox", "arguments": "{}"}} for c in self.last_tool_call_ids]}
        return t
class Tools:
    def set_input_images(self, i): pass
    def dispatch(self, tc, image): return "crop", []
    def cleanup(self): pass
class Main:
    def __init__(self, qj): self.qj, self.calls = list(qj), []
    def judge_good_bad(self, messages, cancel_token=None, **kw):
        last = messages[-1]["content"][0]["text"]
        if "CANDIDATE" in last or "VERIFICATION" in last: self.calls.append("verifier"); s = 0.9
        else: self.calls.append("qj"); s = self.qj.pop(0) if self.qj else 0.9
        return {"score": s, "label": "GOOD" if s >= .5 else "BAD", "raw": "", "log_good": -.1, "log_bad": -3}
def run(cls, script, m):
    cfg = AgentConfig(api_base="x", model_name="m", quality_judge_enabled=True, quality_judge_threshold=0.7, answerability_threshold=0.9, draft_max_steps=8, max_steps=8)
    d = Draft(script); ag = cls(m, d, Tools(), cfg); fb = []
    ag._run_fallback_agent = lambda *a, **k: (fb.append(k.get("trigger")), setattr(a[3], "final_answer", "FB"))
    try: t = ag.run(Image.new("RGB", (32, 32)), Q, sample_id="t")
    finally: ag.shutdown()
    return t, fb, d
ok = True
def check(n, c, d=""):
    global ok; ok &= c; print("PASS" if c else "FAIL", n, d)
check("constant read from env", SA._MAX_CONSECUTIVE_QUALITY_BAD == 1, SA._MAX_CONSECUTIVE_QUALITY_BAD)
m = Main([0.1]); t, fb, d = run(SA.SpeculativeAgentLoop, ["tool", "tool", "tool", "A"], m)
check("async: first BAD -> fallback", fb == ["quality_bad_streak"] and m.calls.count("qj") == 1 and t.final_answer == "FB", f"{fb} {m.calls}")
m = Main([0.9, 0.1]); t, fb, d = run(SA.SpeculativeAgentLoop, ["tool", "tool", "tool", "A"], m)
check("async: GOOD then BAD -> fallback on the BAD", fb == ["quality_bad_streak"] and m.calls.count("qj") == 2, f"{fb} {m.calls}")
m = Main([0.1]); t, fb, d = run(SynchronousSpeculativeAgentLoop, ["tool", "tool", "tool", "A"], m)
check("sync: first BAD -> fallback, only one Draft request", fb == ["quality_bad_streak"] and d.n == 1, f"{fb} draft requests {d.n} {m.calls}")
m = Main([0.9]); t, fb, d = run(SynchronousSpeculativeAgentLoop, ["tool", "A"], m)
check("sync: GOOD steps unaffected", t.final_answer == "A" and not fb, f"{t.final_answer} {fb}")
print("ALL PASSED" if ok else "FAILED")
