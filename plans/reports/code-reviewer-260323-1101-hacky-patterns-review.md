# Code Review: Hacky Patterns in OTel Hook Governance Files

**Branch:** `feat/otel-http-hook-spans`
**Scope:** 12 files, ~5,600 LOC (new + modified)
**Focus:** Redundant state, parameter sprawl, copy-paste, leaky abstractions, stringly-typed code, unnecessary comments

---

## Critical Issues

None found.

## High Priority

### H1. Duplicate `build_auth_headers` functions
- **Category:** Copy-paste with slight variation
- **Files:**
  - `hook_governance.py:131-142` — returns `Authorization`, `User-Agent`, `X-OpenBox-SDK-Version`; hardcodes `_version = "0.1.0"`
  - `client.py:23-32` — returns `Authorization`, `Content-Type`, `User-Agent`; hardcodes version in `User-Agent` string
- **Impact:** Two divergent auth header builders. The hook version includes `X-OpenBox-SDK-Version` but omits `Content-Type`; the client version does the opposite. If headers ever change, one copy will be missed.
- **Recommendation:** Single canonical `build_auth_headers()` in `client.py` (or a shared `_headers.py`), imported everywhere.

### H2. Four near-identical `_build_*_span_data()` functions
- **Category:** Copy-paste with slight variation
- **Files:**
  - `http_governance_hooks.py:74-128` — `_build_http_span_data()`
  - `db_governance_hooks.py:88-156` — `_build_db_span_data()`
  - `file_governance_hooks.py:23-87` — `_build_file_span_data()`
  - `tracing.py:33-74` — `_build_traced_span_data()`
- **Impact:** All four share identical boilerplate: extract span context, get attributes, compute `now_ns / start_time / end_time / duration_ns`, build the common `span_id / trace_id / parent_span_id / name / kind / stage / start_time / end_time / duration_ns / attributes / status / events / hook_type` dict skeleton, then add type-specific fields. ~30 lines of identical code duplicated 4 times.
- **Recommendation:** Extract a `_build_base_span_data(span, name, kind, stage, hook_type, duration_ms, error)` helper in `hook_governance.py` that returns the common dict. Each `_build_*_span_data` calls it and merges type-specific fields.

### H3. Sync/async governance wrappers are nearly identical copy-paste
- **Category:** Copy-paste with slight variation
- **Files:**
  - `db_governance_hooks.py:164-186` (`_evaluate_db_sync`) vs `db_governance_hooks.py:188-209` (`_evaluate_db_async`)
  - `db_governance_hooks.py:252-306` (`_gov_traced_execution`) vs `db_governance_hooks.py:308-358` (`_gov_traced_execution_async`)
  - `db_governance_hooks.py:412-451` (`_asyncpg_governance_wrapper`) duplicates the same started/completed/error pattern from CursorTracer
  - `hook_governance.py:288-332` (`evaluate_sync`) vs `hook_governance.py:334-377` (`evaluate_async`)
  - `tracing.py:156-208` (async wrapper) vs `tracing.py:212-264` (sync wrapper)
  - `http_governance_hooks.py:236-304` (`_httpx_request_hook`) vs `http_governance_hooks.py:359-418` (`_httpx_async_request_hook`)
- **Impact:** The sync and async variants are structurally identical except for `await`. When logic changes, every pair must be updated in lockstep. The DB hooks alone have 3 copies of the "build started SD, call, time, build completed SD, handle error" pattern.
- **Recommendation:** For the governance evaluation pairs (`evaluate_sync`/`evaluate_async`), consider a shared `_evaluate_common()` that builds the payload and checks abort, then has the sync/async caller do the HTTP call. For the DB hooks, a `_wrap_db_operation()` helper accepting the query callable would cut the CursorTracer + asyncpg duplication.

### H4. `_httpx_request_hook` / `_httpx_async_request_hook` near-identical body extraction
- **Category:** Copy-paste with slight variation
- **File:** `http_governance_hooks.py`
  - Lines 251-304 (sync `_httpx_request_hook`)
  - Lines 359-418 (async `_httpx_async_request_hook`)
- **Impact:** The body extraction logic (try `stream._stream`, `stream.body`, `stream._body`, then `_content`, then `content`) is duplicated. The sync version tries `stream._stream` (line 262) but the async version omits it (starts at `stream.body`, line 379). This inconsistency suggests the copy was not kept in sync.
- **Recommendation:** Extract `_extract_httpx_request_body(request) -> Optional[str]` and call it from both hooks.

---

## Medium Priority

### M1. Stringly-typed `on_api_error` parameter
- **Category:** Stringly-typed code
- **Files:** `hook_governance.py:36`, `client.py:57`, `otel_setup.py:86`, `langgraph_handler.py:318`, `config.py:60`
- **Impact:** The error policy is passed as a raw `str` (`"fail_open"` / `"fail_closed"`) across 5+ call sites and compared with string literals. There are already constants `FAIL_OPEN` / `FAIL_CLOSED` in `hook_governance.py:29-30`, but `client.py` and `langgraph_handler.py` compare against raw string literals instead.
- **Recommendation:** Define a `FailPolicy` enum (or at minimum a `Literal["fail_open", "fail_closed"]` type) in `types.py` and use it everywhere. The existing `FAIL_OPEN` / `FAIL_CLOSED` constants in `hook_governance.py` are already half-used; the other modules should import them.

### M2. Stringly-typed `stage` parameter
- **Category:** Stringly-typed code
- **Files:** All `_build_*_span_data()` functions and their callers
- **Impact:** `stage` is always `"started"` or `"completed"` but typed as `str`. A typo like `"complete"` or `"starting"` would silently produce wrong data.
- **Recommendation:** `Literal["started", "completed"]` or a small `Stage` enum.

### M3. Stringly-typed `hook_type` values
- **Category:** Stringly-typed code
- **Files:** `http_governance_hooks.py:118` (`"http_request"`), `db_governance_hooks.py:146` (`"db_query"`), `file_governance_hooks.py:67` (`"file_operation"`), `tracing.py:67` (`"function_call"`)
- **Impact:** Each hook module embeds a magic string. If the server-side contract changes, grep is the only way to find them.
- **Recommendation:** Constants or enum in `types.py`.

### M4. Repeated `import time as _time` inside functions
- **Category:** Unnecessary indirection
- **Files:** `http_governance_hooks.py` (7 occurrences), `file_governance_hooks.py:41`, `tracing.py` (3 occurrences)
- **Impact:** `time` is a stdlib module; importing it at module scope is standard and has no performance cost. Importing inside functions adds noise.
- **Recommendation:** Move `import time` to module scope in all three files.

### M5. `_http_hook_timings` / `_pymongo_pending_commands` / `_redis_span_meta` / `_sa_timings` unbounded growth + blunt clearing
- **Category:** Redundant state / leaky abstraction
- **Files:**
  - `http_governance_hooks.py:38-39` — `_http_hook_timings` dict with `_HTTP_HOOK_TIMINGS_MAX = 1000`, cleared entirely when full
  - `db_governance_hooks.py:48-49` — `_pymongo_pending_commands` with `_PYMONGO_PENDING_MAX = 1000`
  - `db_governance_hooks.py:693-694` — `_redis_span_meta` with `_REDIS_META_MAX = 1000`
  - `db_governance_hooks.py:749` — `_sa_timings` with `_SA_TIMINGS_MAX = 1000`
- **Impact:** The "clear entire dict when cap reached" pattern drops all in-flight timings, corrupting duration calculations for concurrent requests. This is a correctness issue under load.
- **Recommendation:** Use an LRU eviction strategy or `collections.OrderedDict` with FIFO pop, or key expiry. At minimum, pop the oldest N entries instead of clearing all.

### M6. `otel_setup.py` re-exports all private names from sub-modules
- **Category:** Leaky abstraction
- **File:** `otel_setup.py:45-72`
- **Impact:** 20+ private `_`-prefixed names are re-exported "for backward compatibility". This exposes internal implementation details (`_httpx_http_span`, `_http_hook_timings`, `_HTTP_HOOK_TIMINGS_MAX`, `_TEXT_CONTENT_TYPES`) as part of the public surface. Any refactoring of the sub-modules requires maintaining these re-exports indefinitely.
- **Recommendation:** Audit whether any external code actually imports `_build_http_span_data` from `otel_setup`. If only tests do, update the test imports. Remove re-exports of private internals; keep only the public API (`setup_opentelemetry_for_governance`, `uninstrument_all`, etc.).

### M7. `GovernanceBlockedError.__init__` uses positional overloading
- **Category:** Leaky abstraction / parameter sprawl
- **File:** `errors.py:35-62`
- **Impact:** The constructor uses a heuristic (`verdict_or_reason in _HOOK_VERDICTS`) to distinguish "hook-level" vs "SDK-level" calling conventions. This is fragile: if someone passes a reason string that happens to match a verdict name (e.g., `"block"` as a reason), it triggers the wrong branch.
- **Recommendation:** Use a `@classmethod` factory for the hook-level call: `GovernanceBlockedError.from_hook(verdict, reason, identifier)`.

### M8. `_handle_verdict` in `hook_governance.py` duplicates verdict logic
- **Category:** Copy-paste / leaky abstraction
- **File:** `hook_governance.py:237-265`
- **Impact:** This function re-implements verdict string normalization (`lower().replace("-", "_")`) and verdict matching (checking for `"stop"`, `"block"`, `"halt"`, `"require_approval"`, `"request_approval"`) instead of using `Verdict.from_string()` which already handles all of these cases. The codebase has `Verdict` enum with `should_stop()` and `requires_approval()` methods that are ignored here.
- **Recommendation:** Use `Verdict.from_string(data.get("verdict") or data.get("action", "continue"))` then check `.should_stop()` and `.requires_approval()`.

---

## Low Priority

### L1. Unnecessary comments (WHAT, not WHY)
- **Files:** Throughout. Examples:
  - `http_governance_hooks.py:155-156` — `# Hook-level governance evaluation` (the function is called `_requests_request_hook` — obviously a hook)
  - `http_governance_hooks.py:196-197` — `# Hook-level governance evaluation (response stage)` (same)
  - `http_governance_hooks.py:292-293` — `# Store HTTP child span so _patched_send can use it for governance`
  - `http_governance_hooks.py:295-296` — `# Hook-level governance evaluation` (again)
  - `db_governance_hooks.py:267-268` — `# Build & send "started" span data entry`
  - `db_governance_hooks.py:287-288` — `# Build & send "completed" span data entry`
  - `span_processor.py:219-220` — `"""Called when span starts. No-op."""`
  - `file_governance_hooks.py:162-163` — the `with _tracer.start_as_current_span("file.read") as span:` already makes the operation obvious
- **Impact:** Comment noise. The good WHY comments (e.g., `http_governance_hooks.py:21-23` explaining the late import cycle) should be kept.

### L2. `_safe_serialize` in `tracing.py` vs `safe_serialize` in `types.py`
- **Category:** Copy-paste with slight variation
- **Files:** `tracing.py:88-105` vs `types.py:458-483`
- **Impact:** Two serialization functions with overlapping purpose but different signatures and behavior. `_safe_serialize` truncates and returns `str`; `safe_serialize` recursively handles LangChain objects and returns `Any`. Not a bug but contributes to confusion about which to use.

### L3. `_urllib_request_hook` is incomplete
- **Category:** Leaky abstraction
- **File:** `http_governance_hooks.py:748-759`
- **Impact:** The function captures the body into a local variable but never uses it for governance evaluation. There is no `_hook_gov.evaluate_sync()` call, unlike every other request hook.

### L4. `_httpx_response_hook` / `_httpx_async_response_hook` do work but discard results
- **Category:** Redundant state
- **Files:** `http_governance_hooks.py:307-357` and `http_governance_hooks.py:421-479`
- **Impact:** Both response hooks extract body/headers into local variables but then end with a NOTE comment saying governance is handled in `_patched_send`. The extraction work is wasted.

### L5. `GovernanceVerdictResponse` not imported in `langgraph_handler.py`
- **Category:** Potential issue
- **File:** `langgraph_handler.py:99` — uses `GovernanceVerdictResponse` type annotation
- **Impact:** Not in the imports at line 39-46. Works at runtime because of `from __future__ import annotations` (PEP 563 deferred evaluation), but `mypy` may flag it depending on config. The type is used in `_GuardrailsCallbackHandler.__init__` and `_pre_screen_input` return annotation.

---

## Positive Observations

1. **Clean separation of concerns** — hook modules (HTTP/DB/file) are well-isolated from each other
2. **Fail-open semantics** are consistently applied; governance failures don't crash the agent
3. **Thread-safety in `WorkflowSpanProcessor`** — all shared state is behind `self._lock`
4. **`TracedFile` wrapper** is well-designed: proper `__enter__`/`__exit__`, delegates unknown attrs via `__getattr__`
5. **Dedup logic in `GovernanceClient`** prevents duplicate events per activity, with auto-reset on new runs
6. **Exception chain unwrapping** (`_extract_governance_blocked` in `langgraph_handler.py:58-74`) handles LLM SDK wrapping correctly

---

## Summary of Recommended Actions (prioritized)

1. Unify `build_auth_headers` into single source of truth (H1)
2. Extract `_build_base_span_data()` to eliminate 4-way copy-paste (H2)
3. Extract `_extract_httpx_request_body()` to fix sync/async divergence (H4)
4. Replace blunt `dict.clear()` with LRU/FIFO eviction in timing dicts (M5)
5. Add `Literal` or enum types for `stage`, `hook_type`, `on_api_error` (M1-M3)
6. Use `Verdict.from_string()` in `_handle_verdict` instead of manual normalization (M8)
7. Remove dead code in `_httpx_response_hook` and `_urllib_request_hook` (L3, L4)
8. Remove private name re-exports from `otel_setup.py` (M6)

---

## Unresolved Questions

1. Is `_httpx_response_hook` intentionally no-op (governance deferred to `_patched_send`)? If so, why extract body at all?
2. Is `_urllib_request_hook` incomplete (no governance call) by design or oversight?
3. Do any external consumers import private `_`-prefixed names from `otel_setup`?
