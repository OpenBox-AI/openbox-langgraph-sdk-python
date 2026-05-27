"""OpenBox AIP agent identity signing helpers."""

from __future__ import annotations

import base64
import hashlib
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from openbox_langgraph.errors import OpenBoxConfigError

OPENBOX_AGENT_DID_HEADER = "X-OpenBox-Agent-DID"
OPENBOX_AGENT_TIMESTAMP_HEADER = "X-OpenBox-Agent-Timestamp"
OPENBOX_AGENT_NONCE_HEADER = "X-OpenBox-Agent-Nonce"
OPENBOX_BODY_SHA256_HEADER = "X-OpenBox-Body-SHA256"
OPENBOX_AGENT_SIGNATURE_HEADER = "X-OpenBox-Agent-Signature"

_DID_PATTERN = re.compile(
    r"^did:aip:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_ED25519_SEED_BYTE_LENGTH = 32


@dataclass(frozen=True)
class AgentIdentityConfig:
    """Validated AIP identity configuration for one OpenBox agent."""

    did: str
    private_key: str


AgentIdentityHeaders = dict[str, str]


def build_agent_identity_canonical_request(
    *,
    body_sha256: str,
    method: str,
    nonce: str,
    pathname: str,
    timestamp: str,
) -> str:
    """Return Core's newline-joined canonical request string."""
    return "\n".join([method.upper(), pathname, timestamp, nonce, body_sha256])


def validate_agent_identity_config(*, did: str, private_key: str) -> AgentIdentityConfig:
    """Validate and normalize AIP DID/private-key configuration."""
    normalized_did = did.strip()
    normalized_private_key = private_key.strip()

    if not _DID_PATTERN.match(normalized_did):
        msg = "Invalid OpenBox agent DID. Expected format 'did:aip:<uuid>'."
        raise OpenBoxConfigError(msg)

    private_key_seed = _decode_private_key_seed(normalized_private_key)
    return AgentIdentityConfig(
        did=normalized_did,
        private_key=base64.b64encode(private_key_seed).decode("ascii"),
    )


def parse_optional_agent_identity_config(
    *,
    did: str | None,
    private_key: str | None,
) -> AgentIdentityConfig | None:
    """Parse optional DID config and fail fast when only one value is present."""
    normalized_did = _normalize_optional_string(did)
    normalized_private_key = _normalize_optional_string(private_key)

    if normalized_did is None and normalized_private_key is None:
        return None

    if normalized_did is None or normalized_private_key is None:
        msg = (
            "Both OPENBOX_AGENT_DID and OPENBOX_AGENT_PRIVATE_KEY are required "
            "when enabling OpenBox agent identity signing."
        )
        raise OpenBoxConfigError(msg)

    return validate_agent_identity_config(did=normalized_did, private_key=normalized_private_key)


def create_agent_identity_headers(
    *,
    did: str,
    private_key: str,
    method: str,
    pathname: str,
    body: bytes | str | None = None,
    timestamp: str | None = None,
    nonce: str | None = None,
) -> AgentIdentityHeaders:
    """Create signed AIP headers for an exact outbound request body."""
    identity = validate_agent_identity_config(did=did, private_key=private_key)
    body_bytes = _body_to_bytes(body)
    body_sha256 = hashlib.sha256(body_bytes).hexdigest()
    timestamp_value = timestamp or _rfc3339_now()
    nonce_value = nonce or str(uuid.uuid4())
    canonical = build_agent_identity_canonical_request(
        method=method,
        pathname=pathname,
        timestamp=timestamp_value,
        nonce=nonce_value,
        body_sha256=body_sha256,
    )
    signing_key = Ed25519PrivateKey.from_private_bytes(
        _decode_private_key_seed(identity.private_key)
    )
    signature = signing_key.sign(canonical.encode("utf-8"))

    return {
        OPENBOX_AGENT_DID_HEADER: identity.did,
        OPENBOX_AGENT_TIMESTAMP_HEADER: timestamp_value,
        OPENBOX_AGENT_NONCE_HEADER: nonce_value,
        OPENBOX_BODY_SHA256_HEADER: body_sha256,
        OPENBOX_AGENT_SIGNATURE_HEADER: base64.b64encode(signature).decode("ascii"),
    }


def _decode_private_key_seed(private_key: str) -> bytes:
    try:
        decoded = base64.b64decode(private_key, validate=True)
    except ValueError as exc:
        msg = "Invalid OpenBox agent private key. Expected a base64 raw 32-byte Ed25519 seed."
        raise OpenBoxConfigError(msg) from exc

    if (
        len(decoded) != _ED25519_SEED_BYTE_LENGTH
        or base64.b64encode(decoded).decode("ascii") != private_key
    ):
        msg = "Invalid OpenBox agent private key. Expected a base64 raw 32-byte Ed25519 seed."
        raise OpenBoxConfigError(msg)

    return decoded


def _body_to_bytes(body: bytes | str | None) -> bytes:
    if body is None:
        return b""
    if isinstance(body, bytes):
        return body
    return body.encode("utf-8")


def _normalize_optional_string(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _rfc3339_now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
