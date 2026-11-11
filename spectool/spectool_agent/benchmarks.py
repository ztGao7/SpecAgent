"""Per-benchmark runtime profiles.

Profiles are resolved for every sample instead of mutating module globals from
environment variables.  This keeps MMSearch-specific behavior from leaking
into other datasets evaluated by the same Python process.
"""
from pathlib import Path


DEFAULT_PROFILE = "default"
MMSEARCH_PROFILE = "mmsearch"
_KNOWN_PROFILES = {DEFAULT_PROFILE, MMSEARCH_PROFILE}


def is_mmsearch_dataset(dataset: object) -> bool:
    """Return whether a VLMEvalKit dataset name belongs to MMSearch."""
    if dataset is None:
        return False
    normalized = "".join(
        char for char in str(dataset).strip().casefold() if char.isalnum()
    )
    return normalized.startswith("mmsearch")


def resolve_benchmark_profile(
    benchmark_profile: object = None,
    dataset: object = None,
) -> str:
    """Resolve one profile without consulting process-global switches.

    An explicit profile wins.  Otherwise MMSearch is detected from the dataset
    name and every other dataset receives the unchanged default behavior.
    """
    if benchmark_profile is not None:
        profile = str(benchmark_profile).strip().casefold()
        if profile not in _KNOWN_PROFILES:
            known = ", ".join(sorted(_KNOWN_PROFILES))
            raise ValueError(
                f"Unknown benchmark profile {benchmark_profile!r}; expected {known}"
            )
        return profile
    return MMSEARCH_PROFILE if is_mmsearch_dataset(dataset) else DEFAULT_PROFILE


def resolve_sample_id(
    sample_id: object,
    inputs: object,
    benchmark_profile: object = None,
    dataset: object = None,
) -> str:
    """Recover MMSearch's sample id from its image filename when EvalKit omits it.

    VLMEvalKit's API inference path passes the dataset name but not the row id.
    MMSearch images are named with that id, so this targeted fallback restores
    tool-audit attribution without changing identifiers for any other bench.
    """
    if sample_id is not None and str(sample_id).strip():
        return str(sample_id)
    profile = resolve_benchmark_profile(benchmark_profile, dataset)
    if profile != MMSEARCH_PROFILE or not isinstance(inputs, (list, tuple)):
        return "" if sample_id is None else str(sample_id)
    for item in inputs:
        if not isinstance(item, dict) or item.get("type") != "image":
            continue
        value = str(item.get("value") or "").strip()
        if value:
            return Path(value).stem
    return ""
