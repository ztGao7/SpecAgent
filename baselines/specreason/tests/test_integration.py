"""CPU-only tests against frozen EvalKit; no requests to a model or sandbox."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from PIL import Image

from test_adapter import FakeEndpoint, action, answer
from specreason_adapter.core import Config, InfrastructureError
from specreason_adapter.vlmeval_api import SpecReasonAPI
from vlmeval.sample_timing import timed_generate


class IntegrationTests(unittest.TestCase):
    def test_native_tool_result_roundtrip_full_json_and_sample_identity(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {
                "SPECTOOL_IMAGE_TRANSPORT": "data", "SPECTOOL_IMAGE_MAX_SIZE": "0"}):
            image = Path(temp) / "input.png"
            Image.new("RGB", (32, 32), "red").save(image)
            api = SpecReasonAPI(main_model="main", main_url="http://unused/v1",
                draft_model="draft", draft_url="http://unused/v1", raw_dir=temp,
                config=Config())
            api.main = FakeEndpoint("main", scores=[8, 9])
            api.draft = FakeEndpoint("draft", [action("print(1)"), answer()])
            with patch("spectool_agent.tools.ToolRegistry.dispatch", return_value=("1", [])) as dispatch:
                code, prediction, extra = api.generate_inner([
                    {"type": "image", "value": str(image)},
                    {"type": "text", "value": "Which?\nA. yes\nB. no"}],
                    dataset="VStarBench", sample_id=17)
            self.assertEqual(code, 0)
            self.assertEqual(prediction, "A")
            self.assertEqual(dispatch.call_count, 1)
            payload = json.loads(Path(extra["raw_trajectory_path"]).read_text())
            self.assertEqual(payload["sample_id"], "17")
            self.assertTrue(payload["prompts"]["main_equals_draft"])
            self.assertEqual(payload["accounting"]["output_tokens"], 6)
            self.assertEqual(payload["history"][3]["role"], "tool")
            self.assertEqual(payload["history"][3]["tool_call_id"], "c1")
            self.assertIn("cleanup_seconds", payload)
            self.assertEqual(len(payload["input_images"][0]["rgb_sha256"]), 64)

    def test_fatal_failure_is_saved_and_not_swallowed_by_api_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "input.png"
            Image.new("RGB", (32, 32), "red").save(image)
            api = SpecReasonAPI(main_model="main", main_url="http://unused/v1",
                draft_model="draft", draft_url="http://unused/v1", raw_dir=temp)
            api.main = FakeEndpoint("main")
            api.draft = FakeEndpoint("draft", [InfrastructureError("offline-service-test")])
            inputs = [{"type": "image", "value": str(image)}, {"type": "text", "value": "Question"}]
            with self.assertRaises(SystemExit):
                api.generate(message=inputs, dataset="VStarBench", sample_id="fatal")
            saved = list(Path(temp).glob("*.json"))
            self.assertEqual(len(saved), 1)
            self.assertEqual(json.loads(saved[0].read_text())["stop_reason"], "infrastructure_error")
            with self.assertRaises(SystemExit):
                api.generate_inner(inputs, dataset="VStarBench", sample_id="next")

    def test_frozen_outer_timer_passes_index_only_to_opted_in_api(self):
        class Generate:
            requires_sample_id = True

            def generate(self, **kwargs):
                self.seen = kwargs
                return "A"

        with tempfile.TemporaryDirectory() as temp:
            api = Generate()
            result = timed_generate(generate=api.generate, timing_dir=temp, sample_index=42,
                model_name="SpecReason-test", message=[], dataset="VStarBench")
            self.assertEqual(result, "A")
            self.assertEqual(api.seen["sample_id"], 42)
            record = json.loads(next(Path(temp).glob("*.json")).read_text())
            self.assertEqual(record["index"], "42")
            self.assertEqual(record["sample_timing_scope"], "model_generate_call_v1")


if __name__ == "__main__":
    unittest.main()
