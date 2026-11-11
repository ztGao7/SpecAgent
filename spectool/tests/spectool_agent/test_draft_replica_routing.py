import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from vlmeval.api.spectool_speculative import (
    SpeculativeSpectoolAgentAPI,
    _normalize_draft_api_bases,
)


def _routing_api(replica_count=2):
    api = SpeculativeSpectoolAgentAPI.__new__(SpeculativeSpectoolAgentAPI)
    api.draft_models = [f"draft-{idx}" for idx in range(replica_count)]
    api._draft_route_lock = threading.Lock()
    api._draft_route_counter = 0
    return api


def test_normalize_draft_api_bases_accepts_replica_list():
    assert _normalize_draft_api_bases(
        "http://127.0.0.1:8129/v1, http://127.0.0.1:8130/v1/",
        "http://fallback/v1",
    ) == [
        "http://127.0.0.1:8129/v1",
        "http://127.0.0.1:8130/v1",
    ]


def test_api_builds_one_model_per_draft_endpoint(monkeypatch):
    import spectool_agent.draft_model as draft_module
    import spectool_agent.model as model_module

    created_bases = []

    def make_draft(**kwargs):
        created_bases.append(kwargs["api_base"])
        return SimpleNamespace(api_base=kwargs["api_base"])

    monkeypatch.setattr(draft_module, "DraftModel", make_draft)
    monkeypatch.setattr(model_module, "VLLMModel", lambda cfg: MagicMock())

    api = SpeculativeSpectoolAgentAPI(
        model="fake-main",
        api_base="http://main/v1",
        draft_model_name="fake-draft",
        draft_api_base="http://draft-1/v1,http://draft-2/v1",
        agent_repo_root=str(Path(__file__).resolve().parents[2]),
        retry=1,
        verbose=False,
    )

    assert created_bases == ["http://draft-1/v1", "http://draft-2/v1"]
    assert [model.api_base for model in api.draft_models] == created_bases
    assert api.draft_model is api.draft_models[0]


def test_sample_id_routing_is_sticky_and_uses_all_replicas():
    api = _routing_api()
    assignments = {
        sample_id: api._select_draft_model(sample_id)[0]
        for sample_id in range(100)
    }

    assert set(assignments.values()) == {0, 1}
    for sample_id, expected_index in assignments.items():
        assert api._select_draft_model(sample_id)[0] == expected_index


def test_missing_sample_id_uses_thread_safe_round_robin():
    api = _routing_api()

    assert [api._select_draft_model("")[0] for _ in range(5)] == [
        0, 1, 0, 1, 0,
    ]
