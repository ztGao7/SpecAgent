"""CPU check of the ablation switches with scripted Draft/Main."""
import os, sys, time
os.environ.update(SPECTOOL_FAST_ANSWER_ENABLED="0", SPECTOOL_IMAGE_FIRST="assess", SPECTOOL_JUDGE_ANCHOR="fast", SPECTOOL_QJ_ONLY="1")
sys.path.insert(0, sys.argv[1])
from PIL import Image
import spectool_agent.speculative_agent as SA
from spectool_agent.types import AgentConfig, ToolCall
Q = "Which box is red?\nA. left\nB. right\nAnswer with the option letter."
class Draft:
    def __init__(self, script): self.s, self.i = list(script), 0; self.last_truncated = False; self.last_tool_calls = []; self.last_tool_call_ids = []; self.last_assistant_msg = None
    def generate_full(self, msgs, **kw):
        time.sleep(0.3); st = self.s[min(self.i, len(self.s) - 1)]; self.i += 1
        if st == "tool": self.last_tool_calls = [ToolCall("sandbox", {"code": "x"})]; self.last_tool_call_ids = ["c"]; t = "tool"
        else: self.last_tool_calls, self.last_tool_call_ids = [], []; t = f"<answer>{st}</answer>"
        self.last_assistant_msg = {"role": "assistant", "content": t}; return t
class Tools:
    def set_input_images(self, i): pass
    def dispatch(self, tc, image): return "crop", []
    def cleanup(self): pass
class Main:
    def __init__(self, qj, ver): self.qj, self.ver, self.calls = list(qj), ver, []
    def judge_good_bad(self, messages, cancel_token=None, **kw):
        last = messages[-1]["content"][0]["text"]
        if "Answer with GOOD or BAD only." in last and "CANDIDATE" not in last:
            self.calls.append("qj"); s = self.qj.pop(0) if self.qj else 0.9
        else: self.calls.append("verifier"); s = self.ver
        return {"score": s, "label": "GOOD" if s >= .5 else "BAD", "raw": "", "log_good": -.1, "log_bad": -3}
def run(script, m, **env):
    for k in ("SPECTOOL_ABL_NO_STEP_JUDGE", "SPECTOOL_ABL_NO_VERIFIER"): os.environ.pop(k, None)
    os.environ.update(env)
    cfg = AgentConfig(api_base="x", model_name="m", quality_judge_enabled=True, quality_judge_threshold=0.7, answerability_threshold=0.9, draft_max_steps=8, max_steps=8)
    ag = SA.SpeculativeAgentLoop(m, Draft(script), Tools(), cfg); fb = []
    ag._run_fallback_agent = lambda *a, **k: (fb.append(k.get("trigger")), setattr(a[3], "final_answer", "FB"))
    try: t = ag.run(Image.new("RGB", (32, 32)), Q, sample_id="t")
    finally: ag.shutdown()
    return t, fb
ok = True
def check(n, c, d=""):
    global ok; ok &= c; print("PASS" if c else "FAIL", n, d)
m = Main([0.1, 0.1], 0.1); t, fb = run(["tool", "tool", "tool", "A"], m)
check("default: 2 BAD steps -> fallback", fb == ["quality_bad_streak"], f"{fb} {m.calls}")
m = Main([0.1, 0.1], 0.9); t, fb = run(["tool", "tool", "tool", "A"], m, SPECTOOL_ABL_NO_STEP_JUDGE="1")
check("no step judge: no judge request, Draft answer verified", t.final_answer == "A" and "qj" not in m.calls and m.calls == ["verifier"] and not fb, f"{t.final_answer} {m.calls} {fb}")
m = Main([0.9], 0.1); t, fb = run(["tool", "A"], m, SPECTOOL_ABL_NO_VERIFIER="1")
check("no verifier: Draft answer accepted without verifier", t.final_answer == "A" and "verifier" not in m.calls and t.stop_reason == "main_answer_draft_evidence_verified", f"{t.final_answer} {m.calls} {t.stop_reason}")
m = Main([0.9], 0.1); t, fb = run(["tool", "A"], m)
check("default: verifier rejects -> fallback", fb == ["draft_answer_rejected"], f"{fb} {m.calls}")
print("ALL PASSED" if ok else "FAILED")
