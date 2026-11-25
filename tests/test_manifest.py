import json
from pathlib import Path

import pytest
from vllm_hust_ext.manifest import activation_blocker, load_manifest

import lmetric_prefix_router
from lmetric_prefix_router import provider


def test_descriptor_declares_operator_owned_service() -> None:
    manifest = load_manifest(
        Path(lmetric_prefix_router.__file__).with_name("vllm-hust-extension-v0.3.json")
    )
    assert manifest.bundle_id == "io.github.wmaster123.lmetric-prefix-router"
    assert manifest.schema_version == "0.3"
    assert activation_blocker(manifest) is None
    assert len(manifest.requires_services) == 2
    assert manifest.lifecycle_owner == "user"


@pytest.mark.parametrize(
    "wrong",
    [None, "version", "engine_generation", "replay_endpoints", "backend", "ready"],
)
def test_provider_checks_running_service_identity(monkeypatch, wrong):
    manifest = load_manifest(
        Path(lmetric_prefix_router.__file__).with_name("vllm-hust-extension-v0.3.json")
    )
    config = dict(
        router_url="http://127.0.0.1:18201",
        backend_url="http://127.0.0.1:18200",
        engine_generation="a" * 64,
        replay_endpoints={"0": "tcp://127.0.0.1:18400"},
    )
    status = dict(
        version="0.1.0.dev7",
        extension_id=provider.BUNDLE_ID,
        backend=config["backend_url"],
        engine_generation=config["engine_generation"],
        replay_endpoints=config["replay_endpoints"],
        ready=True,
        policy="combined",
    )
    if wrong:
        status[wrong] = False
    monkeypatch.setattr(provider, "version", lambda _: "0.1.0.dev7")
    monkeypatch.setattr(
        provider,
        "get",
        lambda _, path: json.dumps(status) if path == "/status" else b"",
    )
    result = provider.RouterProvider().check(manifest, config)
    assert result.healthy == (wrong is None)
