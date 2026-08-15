"""Anonymous identity for external analytics.

One UUID v4 per install, persisted to disk on first use. The same file
is referenced by ``scripts/install/install.sh`` so install-time beacon
events tie back to the same person across the install→first-run funnel.

No email, no name, no hardware fingerprint — just an opaque UUID.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from openjarvis.core.config import AnalyticsConfig

# Values accepted as "off" / "on" for OPENJARVIS_ANALYTICS. Anything else is
# ignored so a typo falls back to the config rather than silently flipping
# collection on.
_FALSEY = frozenset({"0", "false", "no", "off"})
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def get_or_create_anon_id(path: Path | str) -> str:
    """Return the persisted anon ID, generating one on first call.

    Idempotent across processes — if the file already exists with a
    non-empty value, return it; otherwise generate a fresh UUID v4 and
    write atomically (rename-after-write so a crashed write leaves no
    half-file).
    """
    p = Path(path)
    if p.exists():
        existing = p.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    new_id = str(uuid.uuid4())
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(new_id + "\n", encoding="utf-8")
    tmp.replace(p)
    return new_id


def reset_anon_id(path: Path | str) -> str:
    """Delete the persisted ID and generate a fresh one (privacy reset)."""
    p = Path(path)
    if p.exists():
        p.unlink()
    return get_or_create_anon_id(p)


def is_analytics_enabled(cfg: AnalyticsConfig) -> bool:
    """Return True if external analytics collection is permitted.

    Three inputs, highest precedence first:

    1. ``DO_NOT_TRACK`` — the cross-vendor kill switch. Any value other
       than the falsey set disables collection outright and cannot be
       overridden, so honouring the convention needs no OpenJarvis
       knowledge.
    2. ``OPENJARVIS_ANALYTICS`` — explicit per-run override, ``1``/``0``.
       Unrecognised values are ignored rather than guessed at.
    3. ``cfg.enabled`` — the ``[analytics]`` section of ``config.toml``,
       which defaults to ``False`` in this build.
    """
    dnt = os.environ.get("DO_NOT_TRACK", "").strip().lower()
    if dnt and dnt not in _FALSEY:
        return False

    override = os.environ.get("OPENJARVIS_ANALYTICS", "").strip().lower()
    if override in _FALSEY:
        return False
    if override in _TRUTHY:
        return True

    return cfg.enabled
