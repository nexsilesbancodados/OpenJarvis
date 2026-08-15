"""An engine must never be the Jarvis server itself.

``uzu`` defaults to ``http://localhost:8000`` and ``[server] port`` defaults to
``8000``. Running ``jarvis serve`` on the default port therefore made discovery
probe the Jarvis server, which answers ``/v1/models`` like any
OpenAI-compatible engine — so uzu was reported healthy and inference was routed
back into the process that issued it, surfacing as ``HTTP 405``.
"""

from __future__ import annotations

import pytest

from openjarvis.core.config import JarvisConfig
from openjarvis.engine._discovery import _points_at_own_server


def _config(*, server_host: str = "127.0.0.1", server_port: int = 8000) -> JarvisConfig:
    config = JarvisConfig()
    config.server.host = server_host
    config.server.port = server_port
    return config


class TestSelfDetection:
    def test_the_default_uzu_and_server_ports_collide(self) -> None:
        """The exact shipped defaults, which is how this was found."""
        config = _config()
        config.engine.uzu.host = "http://localhost:8000"
        assert _points_at_own_server("uzu", config) is True

    @pytest.mark.parametrize(
        "host",
        [
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "127.0.0.1:8000",
            "http://[::1]:8000",
        ],
    )
    def test_every_spelling_of_loopback_is_caught(self, host: str) -> None:
        config = _config()
        config.engine.uzu.host = host
        assert _points_at_own_server("uzu", config) is True

    def test_a_different_port_is_fine(self) -> None:
        config = _config()
        config.engine.uzu.host = "http://localhost:8080"
        assert _points_at_own_server("uzu", config) is False

    def test_a_remote_host_on_the_same_port_is_fine(self) -> None:
        """Another machine serving on 8000 is a real engine, not us."""
        config = _config()
        config.engine.vllm.host = "http://192.168.1.50:8000"
        assert _points_at_own_server("vllm", config) is False

    def test_a_non_default_server_port_moves_the_collision(self) -> None:
        config = _config(server_port=9000)
        config.engine.uzu.host = "http://localhost:8000"
        assert _points_at_own_server("uzu", config) is False
        config.engine.uzu.host = "http://localhost:9000"
        assert _points_at_own_server("uzu", config) is True

    @pytest.mark.parametrize("key", ["cloud", "litellm", "gemma_cpp"])
    def test_hostless_engines_are_never_flagged(self, key: str) -> None:
        assert _points_at_own_server(key, _config()) is False

    def test_an_unknown_engine_key_is_not_flagged(self) -> None:
        assert _points_at_own_server("something-new", _config()) is False

    def test_an_empty_host_is_not_flagged(self) -> None:
        config = _config()
        config.engine.uzu.host = ""
        assert _points_at_own_server("uzu", config) is False


class TestDiscoverySkipsIt:
    def test_a_self_pointing_engine_is_never_probed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Probing it would hit our own /v1/models and look healthy."""
        from openjarvis.engine import _discovery

        config = _config()
        config.engine.uzu.host = "http://localhost:8000"
        probed: list[str] = []

        def fake_make(key: str, cfg: JarvisConfig):
            probed.append(key)
            raise RuntimeError("no engine here")

        monkeypatch.setattr(_discovery, "_make_engine", fake_make)
        _discovery.discover_engines(config)
        assert "uzu" not in probed

    def test_get_engine_refuses_it_even_when_asked_by_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from openjarvis.engine import _discovery

        config = _config()
        config.engine.uzu.host = "http://localhost:8000"
        config.engine.default = "uzu"

        def fake_make(key: str, cfg: JarvisConfig):
            raise AssertionError(f"{key} should not have been constructed")

        monkeypatch.setattr(_discovery, "_make_engine", fake_make)
        monkeypatch.setattr(_discovery, "discover_engines", lambda _cfg: [])
        assert _discovery.get_engine(config, "uzu") is None
