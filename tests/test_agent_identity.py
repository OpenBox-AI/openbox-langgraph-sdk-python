"""Tests for OpenBox AIP agent identity signing."""

from __future__ import annotations

import base64
import hashlib

import pytest

from openbox_langgraph.errors import OpenBoxConfigError
from openbox_langgraph.identity import (
    OPENBOX_AGENT_DID_HEADER,
    OPENBOX_AGENT_NONCE_HEADER,
    OPENBOX_AGENT_SIGNATURE_HEADER,
    OPENBOX_AGENT_TIMESTAMP_HEADER,
    OPENBOX_BODY_SHA256_HEADER,
    build_agent_identity_canonical_request,
    create_agent_identity_headers,
    parse_optional_agent_identity_config,
    validate_agent_identity_config,
)

PRIVATE_KEY = base64.b64encode(bytes(range(32))).decode("ascii")
DID = "did:aip:550e8400-e29b-41d4-a716-446655440000"


def test_build_agent_identity_canonical_request() -> None:
    """Canonical request matches Core's newline-joined AIP format."""
    canonical = build_agent_identity_canonical_request(
        method="post",
        pathname="/api/v1/governance/evaluate",
        timestamp="2026-05-21T10:11:12.000Z",
        nonce="nonce-1",
        body_sha256="abc123",
    )

    assert canonical == (
        "POST\n/api/v1/governance/evaluate\n2026-05-21T10:11:12.000Z\nnonce-1\nabc123"
    )


def test_validate_agent_identity_config_normalizes_valid_values() -> None:
    """DID/private-key config is trimmed and private key remains canonical base64."""
    identity = validate_agent_identity_config(
        did=f" {DID} ",
        private_key=f" {PRIVATE_KEY} ",
    )

    assert identity.did == DID
    assert identity.private_key == PRIVATE_KEY


def test_validate_agent_identity_config_rejects_invalid_did() -> None:
    """Invalid DID values fail during config parsing."""
    with pytest.raises(OpenBoxConfigError, match="Invalid OpenBox agent DID"):
        validate_agent_identity_config(did="did:openbox:agent", private_key=PRIVATE_KEY)


def test_validate_agent_identity_config_rejects_invalid_private_key() -> None:
    """Private key must be a canonical base64 raw 32-byte Ed25519 seed."""
    with pytest.raises(OpenBoxConfigError, match="Invalid OpenBox agent private key"):
        validate_agent_identity_config(did=DID, private_key=base64.b64encode(b"short").decode())


def test_parse_optional_agent_identity_config_fails_fast_on_partial_config() -> None:
    """Supplying only one identity env var is a configuration error."""
    message = "Both OPENBOX_AGENT_DID and OPENBOX_AGENT_PRIVATE_KEY"

    with pytest.raises(OpenBoxConfigError, match=message):
        parse_optional_agent_identity_config(did=DID, private_key=None)

    with pytest.raises(OpenBoxConfigError, match=message):
        parse_optional_agent_identity_config(did=None, private_key=PRIVATE_KEY)


def test_create_agent_identity_headers_signs_exact_body_bytes() -> None:
    """AIP headers include the body hash and an Ed25519 signature over the canonical request."""
    body = b'{"foo":"bar"}'
    headers = create_agent_identity_headers(
        did=DID,
        private_key=PRIVATE_KEY,
        method="POST",
        pathname="/api/v1/governance/evaluate",
        body=body,
        timestamp="2026-05-21T10:11:12.000Z",
        nonce="nonce-1",
    )

    assert headers[OPENBOX_AGENT_DID_HEADER] == DID
    assert headers[OPENBOX_AGENT_TIMESTAMP_HEADER] == "2026-05-21T10:11:12.000Z"
    assert headers[OPENBOX_AGENT_NONCE_HEADER] == "nonce-1"
    assert headers[OPENBOX_BODY_SHA256_HEADER] == hashlib.sha256(body).hexdigest()
    assert isinstance(headers[OPENBOX_AGENT_SIGNATURE_HEADER], str)
    assert headers[OPENBOX_AGENT_SIGNATURE_HEADER]
