# openbox/otel_setup.py
"""
Setup OpenTelemetry instrumentors with body capture hooks.

Bodies are stored in the span processor buffer, NOT in OTel span attributes.
This keeps sensitive data out of external tracing systems while still
capturing it for governance evaluation.

Supported HTTP libraries:
- requests
- httpx (sync + async)
- urllib3
- urllib (standard library - request body only)

Supported database libraries:
- psycopg2 (PostgreSQL)
- asyncpg (PostgreSQL async)
- mysql-connector-python
- pymysql
- sqlite3 (SQLite, stdlib)
- pymongo (MongoDB)
- redis
- sqlalchemy (ORM)

``skip_families`` (opt-in hook runtime hybrid exclusivity): when
``use_core_instrumentation=True``, the base ``openbox_core`` runtime installs
its OWN wrappers for the families it covers (http, dbapi, asyncpg,
sqlalchemy, file). Every one of those collides with this module's patches at
the SAME attribute — ``CursorTracer.traced_execution``/``asyncpg.Connection.
execute``/``builtins.open`` are last-writer-wins overwrites (the loser's
governance silently disappears), the OTel HTTP instrumentors are
per-class singletons (a second ``.instrument()`` call is a silent no-op), and
``sqlalchemy.event.listen`` fires EVERY registered listener (both would
evaluate the SAME query twice). ``skip_families`` lets the caller name which
of THIS module's family blocks to skip so the base wrapper is the only one
active for that family — legacy keeps the families base has no coverage for
(urllib3, urllib, pymongo, redis) unconditionally.
"""

import logging
from typing import TYPE_CHECKING, Any, Optional

from . import db_governance_hooks as _db_gov
from . import hook_governance as _hook_gov
from .file_governance_hooks import (
    setup_file_io_instrumentation,
    uninstrument_file_io,
)
from .http_governance_hooks import (
    _httpx_async_request_hook,
    _httpx_async_response_hook,
    _httpx_request_hook,
    _httpx_response_hook,
    _requests_request_hook,
    _requests_response_hook,
    _urllib3_request_hook,
    _urllib3_response_hook,
    _urllib_request_hook,
    setup_httpx_body_capture,
)

if TYPE_CHECKING:
    from .span_processor import WorkflowSpanProcessor

logger = logging.getLogger(__name__)

# Family names ``skip_families`` recognizes — matches the base SDK's own
# ``InstrumentationConfig`` family boundaries exactly (http covers both
# requests+httpx; dbapi covers psycopg2/mysql/pymysql/sqlite3, all funneled
# through the SAME ``CursorTracer.traced_execution`` patch point).
FAMILY_HTTP = "http"
FAMILY_DBAPI = "dbapi"
FAMILY_ASYNCPG = "asyncpg"
FAMILY_SQLALCHEMY = "sqlalchemy"
FAMILY_FILE = "file"

# Global state — hooks in sub-modules reference these via late import of this module
_span_processor: Optional["WorkflowSpanProcessor"] = None
_ignored_url_prefixes: set[str] = set()


def setup_opentelemetry_for_governance(
    span_processor: "WorkflowSpanProcessor",
    api_url: str,
    api_key: str,
    *,
    ignored_urls: list | None = None,
    instrument_databases: bool = True,
    db_libraries: set[str] | None = None,
    instrument_file_io: bool = False,
    sqlalchemy_engine: Any | None = None,
    api_timeout: float = 30.0,
    on_api_error: str = "fail_open",
    agent_did: str | None = None,
    agent_private_key: str | None = None,
    skip_families: set[str] | None = None,
) -> None:
    """
    Setup OpenTelemetry instrumentors with body capture hooks.

    This function instruments HTTP, database, and file I/O libraries to:
    1. Create OTel spans for HTTP requests, database queries, and file operations
    2. Capture request/response bodies (via hooks that store in span_processor)
    3. Register the span processor with the OTel tracer provider

    Args:
        span_processor: The WorkflowSpanProcessor to store bodies in
        ignored_urls: List of URL prefixes to ignore (e.g., OpenBox Core API)
        instrument_databases: Whether to instrument database libraries (default: True)
        db_libraries: Set of database libraries to instrument (None = all available).
                      Valid values: "psycopg2", "asyncpg", "mysql", "pymysql",
                      "pymongo", "redis", "sqlalchemy"
        instrument_file_io: Whether to instrument file I/O operations (default: False)
        sqlalchemy_engine: Optional SQLAlchemy Engine instance to instrument. Required
                          when the engine is created before instrumentation runs (e.g.,
                          at module import time). If not provided, only future engines
                          created via create_engine() will be instrumented.
        agent_did: Optional OpenBox agent DID for AIP request signing.
        agent_private_key: Optional OpenBox agent private key for AIP request signing.
        skip_families: Family names (``FAMILY_HTTP``/``FAMILY_DBAPI``/
                          ``FAMILY_ASYNCPG``/``FAMILY_SQLALCHEMY``/``FAMILY_FILE``)
                          this module must NOT install its own patches for — the
                          base ``openbox_core`` runtime installs those instead
                          (see the module docstring for exactly why running
                          both collides). Default ``None`` skips nothing —
                          identical to every caller that predates this parameter.
    """
    global _span_processor, _ignored_url_prefixes
    _span_processor = span_processor
    families_to_skip = skip_families or set()

    # Set ignored URL prefixes (always include api_url to prevent recursion)
    _ignored_url_prefixes = set(ignored_urls) if ignored_urls else set()
    _ignored_url_prefixes.add(api_url.rstrip("/"))
    logger.info(f"Ignoring URLs with prefixes: {_ignored_url_prefixes}")

    # Configure governance modules
    _hook_gov.configure(
        api_url,
        api_key,
        span_processor,
        api_timeout=api_timeout,
        on_api_error=on_api_error,
        agent_did=agent_did,
        agent_private_key=agent_private_key,
    )
    _db_gov.configure(span_processor)

    # Register span processor with OTel tracer provider
    # This ensures on_end() is called when spans complete
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        # Create a new TracerProvider if none exists
        provider = TracerProvider()
        trace.set_tracer_provider(provider)

    provider.add_span_processor(span_processor)
    logger.info("Registered WorkflowSpanProcessor with OTel TracerProvider")

    # Track what was instrumented
    instrumented = []
    skip_http = FAMILY_HTTP in families_to_skip

    # 1. requests library
    if skip_http:
        logger.info("requests instrumentation skipped — base runtime covers this family")
    else:
        try:
            from opentelemetry.instrumentation.requests import RequestsInstrumentor

            RequestsInstrumentor().instrument(
                request_hook=_requests_request_hook,
                response_hook=_requests_response_hook,
            )
            instrumented.append("requests")
            logger.info("Instrumented: requests")
        except ImportError:
            logger.debug("requests instrumentation not available")

    # 2. httpx library (sync + async) - hooks for metadata only
    if skip_http:
        logger.info("httpx instrumentation skipped — base runtime covers this family")
    else:
        try:
            from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

            HTTPXClientInstrumentor().instrument(
                request_hook=_httpx_request_hook,
                response_hook=_httpx_response_hook,
                async_request_hook=_httpx_async_request_hook,
                async_response_hook=_httpx_async_response_hook,
            )
            instrumented.append("httpx")
            logger.info("Instrumented: httpx")
        except ImportError:
            logger.debug("httpx instrumentation not available")

    # 3. urllib3 library
    try:
        from opentelemetry.instrumentation.urllib3 import URLLib3Instrumentor

        URLLib3Instrumentor().instrument(
            request_hook=_urllib3_request_hook,
            response_hook=_urllib3_response_hook,
        )
        instrumented.append("urllib3")
        logger.info("Instrumented: urllib3")
    except ImportError:
        logger.debug("urllib3 instrumentation not available")

    # 4. urllib (standard library) - request body only, response body cannot be captured
    try:
        from opentelemetry.instrumentation.urllib import URLLibInstrumentor

        URLLibInstrumentor().instrument(
            request_hook=_urllib_request_hook,
        )
        instrumented.append("urllib")
        logger.info("Instrumented: urllib")
    except ImportError:
        logger.debug("urllib instrumentation not available")

    # 5. httpx body capture (separate from OTel - patches Client.send). This
    # ALSO calls `_hook_gov.evaluate_sync`/`evaluate_async` directly — an
    # independent completed-stage evaluation, not merely OTel span metadata —
    # so it must skip alongside the OTel httpx instrumentor above, not just
    # when that block itself fails to import.
    if skip_http:
        logger.info("httpx body capture skipped — base runtime covers this family")
    else:
        setup_httpx_body_capture(span_processor)

    logger.info(f"OpenTelemetry HTTP instrumentation complete. Instrumented: {instrumented}")

    # 6. Database instrumentation (optional)
    if sqlalchemy_engine is not None and not instrument_databases:
        logger.warning(
            "sqlalchemy_engine was provided but instrument_databases=False; "
            "engine will not be instrumented"
        )
    if instrument_databases:
        db_instrumented = setup_database_instrumentation(
            db_libraries, sqlalchemy_engine, families_to_skip
        )
        if db_instrumented:
            instrumented.extend(db_instrumented)

    # 7. File I/O instrumentation (optional). Skipped under FAMILY_FILE: both
    # this module and the base runtime patch `builtins.open` directly —
    # last-writer-wins, so running both silently drops whichever installed
    # first's governance for every file open in the process.
    if instrument_file_io and FAMILY_FILE not in families_to_skip:
        if setup_file_io_instrumentation():
            instrumented.append("file_io")
    elif instrument_file_io:
        logger.info("file I/O instrumentation skipped — base runtime covers this family")

    logger.info(f"OpenTelemetry governance setup complete. Instrumented: {instrumented}")


def setup_database_instrumentation(
    db_libraries: set[str] | None = None,
    sqlalchemy_engine: Any | None = None,
    skip_families: set[str] | None = None,
) -> list[str]:
    """
    Setup OpenTelemetry database instrumentors.

    Database spans will be captured by the WorkflowSpanProcessor (already registered
    with the TracerProvider) and included in governance events.

    Args:
        db_libraries: Set of library names to instrument. If None, instruments all
                      available libraries. Valid values:
                      - "psycopg2" (PostgreSQL sync)
                      - "asyncpg" (PostgreSQL async)
                      - "mysql" (mysql-connector-python)
                      - "pymysql"
                      - "sqlite3" (SQLite, stdlib)
                      - "pymongo" (MongoDB)
                      - "redis"
                      - "sqlalchemy" (ORM)
        sqlalchemy_engine: Optional SQLAlchemy Engine instance to instrument. When
                          provided, registers event listeners on this engine to capture
                          queries. Without this, only engines created after this call
                          (via patched create_engine) will be instrumented.
        skip_families: See ``setup_opentelemetry_for_governance``'s docstring.
                          Per-library OTel span-creation instrumentors (which
                          base never touches) still install even when their
                          family is skipped — only the GOVERNANCE-EVALUATING
                          hook installer for that family is skipped, since
                          that installer is what collides with base's own
                          wrapper for the SAME attribute/event.

    Returns:
        List of successfully instrumented library names
    """
    instrumented = []
    families_to_skip = skip_families or set()

    # ── pymongo CommandListener first (must register before MongoClient creation) ──
    if db_libraries is None or "pymongo" in db_libraries:
        _db_gov.setup_pymongo_hooks()

    # ── OTel dbapi instrumentors (governance via CursorTracer patch below) ──
    if db_libraries is None or "psycopg2" in db_libraries:
        try:
            from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
            Psycopg2Instrumentor().instrument()
            instrumented.append("psycopg2")
            logger.info("Instrumented: psycopg2")
        except ImportError:
            logger.debug("psycopg2 OTel instrumentation not available")

    if db_libraries is None or "asyncpg" in db_libraries:
        try:
            from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
            AsyncPGInstrumentor().instrument()
            instrumented.append("asyncpg")
            logger.info("Instrumented: asyncpg")
        except ImportError:
            logger.debug("asyncpg OTel instrumentation not available")

    if db_libraries is None or "mysql" in db_libraries:
        try:
            from opentelemetry.instrumentation.mysql import MySQLInstrumentor
            MySQLInstrumentor().instrument()
            instrumented.append("mysql")
            logger.info("Instrumented: mysql")
        except ImportError:
            logger.debug("mysql OTel instrumentation not available")

    if db_libraries is None or "pymysql" in db_libraries:
        try:
            from opentelemetry.instrumentation.pymysql import PyMySQLInstrumentor
            PyMySQLInstrumentor().instrument()
            instrumented.append("pymysql")
            logger.info("Instrumented: pymysql")
        except ImportError:
            logger.debug("pymysql OTel instrumentation not available")

    if db_libraries is None or "sqlite3" in db_libraries:
        try:
            from opentelemetry.instrumentation.sqlite3 import SQLite3Instrumentor
            SQLite3Instrumentor().instrument()
            instrumented.append("sqlite3")
            logger.info("Instrumented: sqlite3")
        except ImportError:
            logger.debug("sqlite3 OTel instrumentation not available")

    # pymongo OTel (CommandListener already registered above)
    if db_libraries is None or "pymongo" in db_libraries:
        try:
            from opentelemetry.instrumentation.pymongo import PymongoInstrumentor
            PymongoInstrumentor().instrument()
            instrumented.append("pymongo")
            logger.info("Instrumented: pymongo")
        except ImportError:
            logger.debug("pymongo OTel instrumentation not available")

    # redis — pass governance hooks to OTel instrumentor (native support)
    if db_libraries is None or "redis" in db_libraries:
        try:
            from opentelemetry.instrumentation.redis import RedisInstrumentor

            req_hook, resp_hook = _db_gov.setup_redis_hooks()
            RedisInstrumentor().instrument(
                request_hook=req_hook, response_hook=resp_hook,
            )
            instrumented.append("redis")
            logger.info("Instrumented: redis")
        except ImportError:
            logger.debug("redis instrumentation not available")

    # sqlalchemy (ORM)
    if (
        sqlalchemy_engine is not None
        and db_libraries is not None
        and "sqlalchemy" not in db_libraries
    ):
        logger.warning(
            "sqlalchemy_engine was provided but 'sqlalchemy' is not in db_libraries; "
            "engine will not be instrumented"
        )
    if db_libraries is None or "sqlalchemy" in db_libraries:
        try:
            from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

            if sqlalchemy_engine is not None:
                # Validate engine type before passing to instrumentor
                try:
                    from sqlalchemy.engine import Engine as _SAEngine
                except ImportError as exc:
                    raise TypeError(
                        "sqlalchemy_engine was provided but sqlalchemy is not installed"
                    ) from exc
                if not isinstance(sqlalchemy_engine, _SAEngine):
                    raise TypeError(
                        f"sqlalchemy_engine must be a sqlalchemy.engine.Engine instance, "
                        f"got {type(sqlalchemy_engine).__name__}"
                    )
                # Governance hooks on engine events — skipped when the base
                # runtime's OWN class-level `event.listen(Engine, ...)` covers
                # this family (both would fire for this engine otherwise;
                # `SQLAlchemyInstrumentor` below is pure OTel span metadata,
                # never a governance-evaluate call, so it stays unconditional).
                if FAMILY_SQLALCHEMY not in families_to_skip:
                    _db_gov.setup_sqlalchemy_hooks(sqlalchemy_engine)
                # Instrument the existing engine directly (registers event listeners)
                SQLAlchemyInstrumentor().instrument(engine=sqlalchemy_engine)
                logger.info("Instrumented: sqlalchemy (existing engine)")
            else:
                # Patch create_engine() for future engines only
                SQLAlchemyInstrumentor().instrument()
                logger.info("Instrumented: sqlalchemy (future engines)")
            instrumented.append("sqlalchemy")
        except ImportError:
            logger.debug("sqlalchemy instrumentation not available")

    # ── Governance hooks for dbapi libs (must be AFTER instrumentors) ──
    # OTel dbapi instrumentors silently discard request_hook/response_hook kwargs.
    # Instead, we patch CursorTracer.traced_execution to inject governance hooks
    # around the query_method call (runs inside the OTel span context).
    # Skipped under FAMILY_DBAPI: base patches the SAME
    # `CursorTracer.traced_execution` attribute — whichever installs LAST would
    # silently overwrite the other's governance for psycopg2/mysql/pymysql/sqlite3.
    dbapi_libs = {"psycopg2", "mysql", "pymysql", "sqlite3"}
    if FAMILY_DBAPI in families_to_skip:
        logger.info(
            "CursorTracer governance hooks skipped — base runtime covers dbapi libs"
        )
    elif any(lib in instrumented for lib in dbapi_libs):
        if _db_gov.install_cursor_tracer_hooks():
            logger.info("CursorTracer governance hooks installed for dbapi libs")

    # asyncpg uses its own _do_execute (not CursorTracer) — needs separate wrapt
    # hooks. Skipped under FAMILY_ASYNCPG: base wraps `Connection._execute`
    # directly, and legacy's wrapt wrapper sits on the PUBLIC `Connection.execute`
    # (which calls `_execute` internally) — both active means every asyncpg
    # query is evaluated TWICE, once per wrapper layer.
    if "asyncpg" in instrumented:
        if FAMILY_ASYNCPG in families_to_skip:
            logger.info("asyncpg governance hooks skipped — base runtime covers this family")
        else:
            _db_gov.install_asyncpg_hooks()

    if instrumented:
        logger.info(f"Database instrumentation complete. Instrumented: {instrumented}")
    else:
        logger.debug("No database libraries instrumented (none available or installed)")

    return instrumented


def uninstrument_databases() -> None:
    """Uninstrument all database libraries."""
    try:
        from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor

        Psycopg2Instrumentor().uninstrument()
    except (ImportError, Exception):
        pass

    try:
        from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor

        AsyncPGInstrumentor().uninstrument()
    except (ImportError, Exception):
        pass

    try:
        from opentelemetry.instrumentation.mysql import MySQLInstrumentor

        MySQLInstrumentor().uninstrument()
    except (ImportError, Exception):
        pass

    try:
        from opentelemetry.instrumentation.pymysql import PyMySQLInstrumentor

        PyMySQLInstrumentor().uninstrument()
    except (ImportError, Exception):
        pass

    try:
        from opentelemetry.instrumentation.sqlite3 import SQLite3Instrumentor

        SQLite3Instrumentor().uninstrument()
    except (ImportError, Exception):
        pass

    try:
        from opentelemetry.instrumentation.pymongo import PymongoInstrumentor

        PymongoInstrumentor().uninstrument()
    except (ImportError, Exception):
        pass

    try:
        from opentelemetry.instrumentation.redis import RedisInstrumentor

        RedisInstrumentor().uninstrument()
    except (ImportError, Exception):
        pass

    try:
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

        SQLAlchemyInstrumentor().uninstrument()
    except (ImportError, Exception):
        pass

    # Clean up DB governance hooks
    _db_gov.uninstrument_all()


def uninstrument_all() -> None:
    """Uninstrument all HTTP and database libraries."""
    global _span_processor, _ignored_url_prefixes
    _span_processor = None
    _ignored_url_prefixes = set()

    # Uninstrument HTTP libraries
    try:
        from opentelemetry.instrumentation.requests import RequestsInstrumentor

        RequestsInstrumentor().uninstrument()
    except (ImportError, Exception):
        pass

    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        HTTPXClientInstrumentor().uninstrument()
    except (ImportError, Exception):
        pass

    try:
        from opentelemetry.instrumentation.urllib3 import URLLib3Instrumentor

        URLLib3Instrumentor().uninstrument()
    except (ImportError, Exception):
        pass

    try:
        from opentelemetry.instrumentation.urllib import URLLibInstrumentor

        URLLibInstrumentor().uninstrument()
    except (ImportError, Exception):
        pass

    # Uninstrument database libraries
    uninstrument_databases()

    # Uninstrument file I/O
    uninstrument_file_io()
