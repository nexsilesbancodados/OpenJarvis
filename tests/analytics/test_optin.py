"""Tests for the external-analytics opt-in gate.

This build ships session-awareness observers that read window titles,
terminal commands and repository state, so external collection must stay
off unless someone turns it on deliberately. These tests pin that
default and the two environment overrides.
"""

from __future__ import annotations

import pytest

from openjarvis.analytics.identity import is_analytics_enabled
from openjarvis.core.config import AnalyticsConfig


@pytest.fixture(autouse=True)
def _clear_analytics_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralise the ambient environment so tests assert the gate itself."""
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    monkeypatch.delenv("OPENJARVIS_ANALYTICS", raising=False)


class TestDefault:
    def test_disabled_by_default(self) -> None:
        """A freshly constructed config must not permit collection."""
        assert AnalyticsConfig().enabled is False
        assert is_analytics_enabled(AnalyticsConfig()) is False

    def test_config_can_enable(self) -> None:
        assert is_analytics_enabled(AnalyticsConfig(enabled=True)) is True


class TestDoNotTrack:
    @pytest.mark.parametrize("value", ["1", "true", "yes", "on", "anything"])
    def test_overrides_enabled_config(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        """DO_NOT_TRACK wins even when the config says yes."""
        monkeypatch.setenv("DO_NOT_TRACK", value)
        assert is_analytics_enabled(AnalyticsConfig(enabled=True)) is False

    def test_beats_opposing_openjarvis_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """DO_NOT_TRACK outranks an explicit OPENJARVIS_ANALYTICS=1."""
        monkeypatch.setenv("DO_NOT_TRACK", "1")
        monkeypatch.setenv("OPENJARVIS_ANALYTICS", "1")
        assert is_analytics_enabled(AnalyticsConfig(enabled=True)) is False

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
    def test_falsey_does_not_disable(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        """DO_NOT_TRACK=0 means "no preference", not "force off"."""
        monkeypatch.setenv("DO_NOT_TRACK", value)
        assert is_analytics_enabled(AnalyticsConfig(enabled=True)) is True


class TestEnvOverride:
    @pytest.mark.parametrize("value", ["1", "true", "yes", "on", "ON", " 1 "])
    def test_enables_over_disabled_config(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("OPENJARVIS_ANALYTICS", value)
        assert is_analytics_enabled(AnalyticsConfig(enabled=False)) is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF"])
    def test_disables_over_enabled_config(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("OPENJARVIS_ANALYTICS", value)
        assert is_analytics_enabled(AnalyticsConfig(enabled=True)) is False

    @pytest.mark.parametrize("value", ["maybe", "2", "yep"])
    def test_unrecognised_value_falls_through_to_config(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        """A typo must not be read as consent."""
        monkeypatch.setenv("OPENJARVIS_ANALYTICS", value)
        assert is_analytics_enabled(AnalyticsConfig(enabled=False)) is False
        assert is_analytics_enabled(AnalyticsConfig(enabled=True)) is True
