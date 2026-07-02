"""initialize() preserves URL-security + strict API-key-format validation.

Layered env precedence (explicit > OPENBOX_LANGGRAPH_* > OPENBOX_*) is exercised
on the runtime builder in test_core_runtime_config.py; initialize()'s api_url /
api_key are required arguments, so the guarantees to pin here are the security
validations, which must survive the base-SDK integration.
"""

from __future__ import annotations

import pytest

from openbox_langgraph.config import initialize
from openbox_langgraph.errors import OpenBoxAuthError, OpenBoxInsecureURLError


def test_rejects_non_localhost_http() -> None:
    with pytest.raises(OpenBoxInsecureURLError):
        initialize(api_url="http://evil.example.com", api_key="obx_test_abc", validate=False)


def test_allows_localhost_http() -> None:
    # localhost over http is allowed; validate=False skips the server round-trip.
    initialize(api_url="http://localhost:8080", api_key="obx_test_abc", validate=False)


def test_rejects_malformed_api_key() -> None:
    with pytest.raises(OpenBoxAuthError):
        initialize(api_url="https://core.openbox.ai", api_key="not-a-valid-key", validate=False)
