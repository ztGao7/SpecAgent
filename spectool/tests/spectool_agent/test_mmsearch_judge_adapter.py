"""MMSearch Judge must see only the final report, never EvalKit trajectory data."""
import importlib.util
from pathlib import Path


def _load_scorer():
    path = Path(__file__).resolve().parents[2] / "scripts" / "score_mmsearch_vdr_judge.py"
    spec = importlib.util.spec_from_file_location("mmsearch_vdr_judge", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_unwraps_python_repr_and_answer_tag():
    scorer = _load_scorer()
    wrapped = {
        "prediction": "<answer>The supported final report.</answer>",
        "extra_records": {"trajectory": "ground truth must not leak"},
    }
    inner = scorer._unwrap_evalkit_prediction(str(wrapped))
    assert scorer._extract_final_answer(inner) == "The supported final report."


def test_unwraps_prediction_from_truncated_wrapper():
    scorer = _load_scorer()
    wrapped = (
        "{'prediction': '<answer>Final report with \\\'quoted\\\' evidence.</answer>', "
        "'extra_records': {'trajectory': '" + "x" * 40000
    )
    inner = scorer._unwrap_evalkit_prediction(wrapped[:32767])
    assert scorer._extract_final_answer(inner) == "Final report with 'quoted' evidence."


def test_plain_report_is_unchanged():
    scorer = _load_scorer()
    report = "A plain final report."
    assert scorer._unwrap_evalkit_prediction(report) == report
