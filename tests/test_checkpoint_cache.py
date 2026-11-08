"""Named uncertainty weights refresh without losing a usable cached copy."""

from io import BytesIO
import json
from urllib.error import HTTPError, URLError

import pytest

from helio_foxes import api


def response(payload, **headers):
    stream = BytesIO(payload)
    stream.headers = {"Content-Length": str(len(payload)), **headers}
    return stream


@pytest.fixture
def cache(monkeypatch, tmp_path):
    monkeypatch.setenv("HELIO_FOXES_CACHE", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_changed_weights_replace_cache_and_subsequent_load_revalidates(cache, monkeypatch):
    requests = []

    def urlopen(request, timeout):
        assert timeout == 10
        headers = {} if isinstance(request, str) else {k.lower(): v for k, v in request.header_items()}
        requests.append(headers)
        if len(requests) == 1:
            return response(b"old weights", ETag='"old"')
        if len(requests) == 2:
            assert headers["if-none-match"] == '"old"'
            return response(b"new weights", ETag='"new"')
        assert headers["if-none-match"] == '"new"'
        raise HTTPError(request.full_url, 304, "Not Modified", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    target = api._resolve_checkpoint("uncertainty")
    assert target.read_bytes() == b"old weights"
    assert api._resolve_checkpoint("uncertainty") == target
    assert target.read_bytes() == b"new weights"
    assert api._resolve_checkpoint("uncertainty") == target
    assert target.read_bytes() == b"new weights"
    metadata = json.loads((cache / "UncertaintyModel.ckpt.http.json").read_text())
    assert metadata["etag"] == '"new"'
    assert not list(cache.glob("*.part"))


@pytest.mark.parametrize("metadata", [None, "{", "[]"])
def test_older_or_invalid_cache_metadata_is_refreshed_once(cache, monkeypatch, metadata):
    target = cache / "UncertaintyModel.ckpt"
    target.write_bytes(b"old weights")
    if metadata is not None:
        target.with_name(target.name + ".http.json").write_text(metadata)

    def urlopen(request, timeout):
        assert isinstance(request, str)
        return response(b"new weights", ETag='"new"')

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    assert api._resolve_checkpoint("uncertainty").read_bytes() == b"new weights"


def test_last_modified_validator_is_used_when_etag_is_unavailable(cache, monkeypatch):
    modified = "Sun, 08 Nov 2026 12:00:00 GMT"
    requests = []

    def urlopen(request, timeout):
        requests.append(request)
        if len(requests) == 1:
            return response(b"weights", **{"Last-Modified": modified})
        headers = {k.lower(): v for k, v in request.header_items()}
        assert headers["if-modified-since"] == modified
        raise HTTPError(request.full_url, 304, "Not Modified", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    target = api._resolve_checkpoint("uncertainty")
    assert api._resolve_checkpoint("uncertainty") == target
    assert target.read_bytes() == b"weights"


def test_matching_etag_skips_body_even_when_server_returns_200(cache, monkeypatch):
    target = api._resolve_checkpoint
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: response(b"weights", ETag='"same"'))
    path = target("uncertainty")
    unchanged = response(b"body must not be downloaded", ETag='"same"')
    monkeypatch.setattr(unchanged, "read", lambda *a: pytest.fail("Unchanged checkpoint body was read"))
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: unchanged)
    assert target("uncertainty") == path
    assert unchanged.closed
    assert path.read_bytes() == b"weights"


@pytest.mark.parametrize("error", [URLError("offline"), TimeoutError("timed out")])
def test_unreachable_server_reuses_existing_cache_with_warning(cache, monkeypatch, caplog, error):
    target = cache / "UncertaintyModel.ckpt"
    target.write_bytes(b"cached weights")

    def urlopen(*args, **kwargs):
        raise error

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    assert api._resolve_checkpoint("uncertainty") == target
    assert target.read_bytes() == b"cached weights"
    assert "using cached weights" in caplog.text
    assert not list(cache.glob("*.part"))


@pytest.mark.parametrize("payload,headers,reason", [
    (b"version https://git-lfs.github.com/spec/v1\n", {}, "LFS pointer"),
    (b"short", {"Content-Length": "100"}, "Incomplete checkpoint"),
    (b"", {}, "Incomplete checkpoint"),
])
def test_bad_refresh_preserves_existing_weights_and_metadata(cache, monkeypatch, payload, headers, reason):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: response(b"good weights", ETag='"old"'))
    target = api._resolve_checkpoint("uncertainty")
    metadata = target.with_name(target.name + ".http.json")
    previous_metadata = metadata.read_bytes()
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: response(payload, ETag='"new"', **headers))
    with pytest.raises(RuntimeError, match=reason):
        api._resolve_checkpoint("uncertainty")
    assert target.read_bytes() == b"good weights"
    assert metadata.read_bytes() == previous_metadata
    assert not list(cache.glob("*.part"))


def test_missing_cache_and_unreachable_server_raises(cache, monkeypatch):
    def urlopen(*args, **kwargs):
        raise URLError("offline")

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    with pytest.raises(URLError, match="offline"):
        api._resolve_checkpoint("uncertainty")
    assert not list(cache.iterdir())


def test_local_paths_and_cached_legacy_names_do_not_contact_server(cache, monkeypatch):
    def urlopen(*args, **kwargs):
        pytest.fail("Local and legacy cached weights must not check the network")

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    local = cache / "uncertainty"
    local.write_bytes(b"local weights")
    assert api._resolve_checkpoint("uncertainty") == local
    legacy = cache / "LegacyModels/no-attention-mask.ckpt"
    legacy.parent.mkdir()
    legacy.write_bytes(b"legacy weights")
    assert api._resolve_checkpoint("no-attention-mask") == legacy
