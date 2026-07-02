"""Volatile-field normalization for golden wire-payload fixtures.

Fixtures must be byte-reproducible across regenerations so a future parity
gate can diff generated output against them. Timestamps, nonces, signatures,
and generated ids are volatile per-run and must resolve to the same stable
placeholder every time — this module is the single source of truth for that
normalization so `write_fixture_pair` and `write_ordering_fixture` (in
capture_harness.py) stay in sync.
"""

from __future__ import annotations

import re
from typing import Any

# Placeholders substituted for volatile fields at normalization time.
_TIMESTAMP_PLACEHOLDER = "<TS>"
_NONCE_PLACEHOLDER = "<NONCE>"
_ID_PLACEHOLDER = "<ID>"
_DURATION_PLACEHOLDER = "<DURATION_MS>"

# Wall-clock timing fields on LangChainGovernanceEvent (types.py: duration_ms,
# start_time, end_time) are real elapsed-time floats — every real run produces
# a different value by definition. Unlike ids/timestamps/signatures, a float's
# VALUE gives no reliable shape signal that it's a timing measurement (any
# float could be genuine data), so this one exception normalizes by key name.
_VOLATILE_NUMERIC_KEYS = frozenset({"duration_ms", "start_time", "end_time"})

# RFC3339 timestamp with milliseconds, e.g. 2026-07-02T10:00:00.000Z
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z$")

# The literal suffixes the handler appends to a generated workflow_id/run_id to
# build an activity_id (langgraph_handler.py: f"{run_id}-wf", f"{run_id}-pre",
# f"{run_id}-sig", f"{run_id}-pre-c", event_run_id + "-c"). Captured so the
# placeholder preserves suffix semantics, e.g. "<ID>-pre-c" not just "<ID>".
_ID_SUFFIXES = ("-pre-c", "-pre", "-wf", "-sig", "-c")
_ID_SUFFIX_PATTERN = r"(?:-pre-c|-pre|-wf|-sig|-c)?"

# Full canonical UUID (v4 nonces, and LangGraph's bare uuid7 event_run_id used
# directly as an activity_id), with an optional handler-appended suffix.
_BARE_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}" + _ID_SUFFIX_PATTERN + r"$",
    re.IGNORECASE,
)
# The handler's own generated id scheme: `{thread_id}-{uuid4().hex[:8]}` for
# workflow_id and `{thread_id}-run-{uuid4().hex[8:16]}` for run_id (see
# OpenBoxLangGraphHandler.ainvoke), with an optional handler-appended suffix.
# `thread_id` is caller-supplied and arbitrary, so we anchor on the trailing
# "(-run-)?{8 hex chars}(suffix)?" shape rather than a fixed prefix — this is
# specific enough to avoid matching hand-authored literal fixture ids like
# "tool-run-1" or "chain-run-1" (those end in decimal digits, not 8 hex chars).
_GENERATED_ID_RE = re.compile(
    r"^.+-(?:run-)?[0-9a-f]{8}" + _ID_SUFFIX_PATTERN + r"$", re.IGNORECASE
)
# A canonical UUID appearing ANYWHERE inside a longer string — e.g. LangChain's
# own AIMessage.id field, shaped "lc_run--{uuid7}-0". Unanchored (no ^/$) so it
# matches as a substring; used only as a fallback after the full-string checks
# above have already failed, to normalize ids the SDK itself never generates
# but that still land in a captured wire body via a nested LangChain object.
_EMBEDDED_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)


def normalize_json(value: Any) -> Any:
    """Recursively replace volatile field values with stable placeholders.

    Volatile values are RFC3339 timestamps; UUID4 nonces and LangGraph's bare
    uuid7 event_run_ids; the handler's own generated workflow_id/run_id
    scheme (`{thread_id}-{hex8}`, `{thread_id}-run-{hex8}`) and their
    `-wf`/`-pre`/`-sig`/`-c`/`-pre-c` suffixed activity_id derivatives;
    Ed25519 signatures / SHA256 body digests; and wall-clock timing floats
    (`duration_ms`, `start_time`, `end_time` — the one field class normalized
    by KEY NAME, since a duration's numeric value has no shape signal). We
    otherwise normalize by VALUE SHAPE (regex) rather than by key name so
    nested/renamed fields still resolve to the same placeholder after a
    refactor. A matched literal suffix is preserved on the id placeholder
    (e.g. "<ID>-pre-c") so suffix semantics — which row an activity_id
    refers to — stay visible in the fixture.
    """
    if isinstance(value, dict):
        return {
            k: (_DURATION_PLACEHOLDER if k in _VOLATILE_NUMERIC_KEYS and v is not None
                else normalize_json(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [normalize_json(v) for v in value]
    if isinstance(value, str):
        if _TIMESTAMP_RE.match(value):
            return _TIMESTAMP_PLACEHOLDER
        generated_id = _normalize_generated_id(value)
        if generated_id is not None:
            return generated_id
        # Ed25519 signatures and body-sha256 hex digests are also volatile
        # (they depend on the timestamp/nonce baked into the canonical
        # request) — normalize base64 signature-shaped and hex-64 values.
        if _looks_like_signature_or_hash(value):
            return _NONCE_PLACEHOLDER
        # Fallback: a UUID embedded as a substring of a longer string (e.g.
        # LangChain's own AIMessage.id = "lc_run--{uuid7}-0"), which the
        # full-string checks above never match by design.
        if _EMBEDDED_UUID_RE.search(value):
            return _EMBEDDED_UUID_RE.sub(_ID_PLACEHOLDER, value)
        return value
    return value


def _normalize_generated_id(value: str) -> str | None:
    """Return `<ID>` (or `<ID>{suffix}`) if `value` looks like a generated id.

    Matches both a bare canonical UUID (nonces, LangGraph's raw event_run_id)
    and the handler's own `{prefix}-(run-)?{hex8}` scheme. Returns None (no
    match) for hand-authored literal fixture ids like "tool-run-1", which end
    in decimal digits rather than 8 hex characters.
    """
    for pattern in (_BARE_UUID_RE, _GENERATED_ID_RE):
        if pattern.match(value):
            return f"{_ID_PLACEHOLDER}{_extract_suffix(value)}"
    return None


def _extract_suffix(value: str) -> str:
    """Return the trailing handler-appended suffix literal, or "" if none."""
    for suffix in _ID_SUFFIXES:
        if value.endswith(suffix):
            return suffix
    return ""


def _looks_like_signature_or_hash(value: str) -> bool:
    """Heuristic: 64-hex-char SHA256 digest or 88-char base64 Ed25519 signature."""
    if re.match(r"^[0-9a-f]{64}$", value):
        return True
    return bool(re.match(r"^[A-Za-z0-9+/]{86}==$", value))
