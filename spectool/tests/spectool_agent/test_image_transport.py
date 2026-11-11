import base64
import io
from pathlib import Path
from urllib.parse import unquote, urlparse

from PIL import Image

from spectool_agent.model import downscale_data_url, pil_to_data_url


def _open_url(url: str) -> Image.Image:
    if url.startswith("data:image/"):
        return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    return Image.open(unquote(urlparse(url).path))


def test_data_transport_applies_area_cap_without_mutating_source(monkeypatch):
    monkeypatch.setenv("SPECTOOL_IMAGE_TRANSPORT", "data")
    monkeypatch.setenv("SPECTOOL_IMAGE_MAX_PIXELS", "4096")
    monkeypatch.setenv("SPECTOOL_IMAGE_RESIZE_FACTOR", "8")
    source = Image.new("RGB", (128, 128), "white")

    url = pil_to_data_url(source, max_size=0)

    assert source.size == (128, 128)
    with _open_url(url) as encoded:
        assert encoded.size == (64, 64)


def test_file_transport_reuses_content_addressed_media(monkeypatch, tmp_path):
    monkeypatch.setenv("SPECTOOL_IMAGE_TRANSPORT", "file")
    monkeypatch.setenv("SPECTOOL_LOCAL_MEDIA_ROOT", str(tmp_path))
    monkeypatch.setenv("SPECTOOL_IMAGE_MAX_PIXELS", "4096")
    monkeypatch.setenv("SPECTOOL_IMAGE_RESIZE_FACTOR", "8")
    source = Image.new("RGB", (128, 128), "white")

    first = pil_to_data_url(source, max_size=0)
    second = pil_to_data_url(source, max_size=0)

    assert first == second
    assert first.startswith("file://")
    assert len(list(Path(tmp_path).glob("*.jpg"))) == 1
    with _open_url(first) as encoded:
        assert encoded.size == (64, 64)


def test_draft_downscale_supports_file_transport(monkeypatch, tmp_path):
    monkeypatch.setenv("SPECTOOL_IMAGE_TRANSPORT", "file")
    monkeypatch.setenv("SPECTOOL_LOCAL_MEDIA_ROOT", str(tmp_path))
    monkeypatch.setenv("SPECTOOL_IMAGE_MAX_PIXELS", "0")
    monkeypatch.setenv("SPECTOOL_IMAGE_RESIZE_FACTOR", "1")
    source_url = pil_to_data_url(Image.new("RGB", (100, 50)), max_size=0)

    resized_url = downscale_data_url(source_url, max_size=40)

    with _open_url(resized_url) as encoded:
        assert encoded.size == (40, 20)
