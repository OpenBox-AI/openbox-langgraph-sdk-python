# Code Reuse Review: feat/otel-http-hook-spans

**Date:** 2026-03-23
**Branch:** feat/otel-http-hook-spans
**Reviewer:** code-reviewer

---

## Scope

- Files: 12 changed/new source files
- Focus: Duplicate patterns across OTel hook governance modules
- Key new files: http_governance_hooks.py (759 LOC), db_governance_hooks.py (874 LOC), file_governance_hooks.py (407 LOC), hook_governance.py (377 LOC), tracing.py (320 LOC)

---

## Finding 1: CRITICAL — Duplicate `build_auth_headers()` (2 independent implementations)

| Location | Signature | Headers Produced |
|---|---|---|
| `client.py:23` | `build_auth_headers(api_key) -> dict` | `Authorization`, `Content-Type`, `User-Agent` (OpenBox-LangGraph-SDK/0.1.0) |
| `hook_governance.py:131` | `build_auth_headers(api_key) -> dict` | `Authorization`, `User-Agent` (OpenBox-SDK/0.1.0), `X-OpenBox-SDK-Version` |

**Impact:** The two functions produce *different* header sets. `client.py` includes `Content-Type: application/json`; `hook_governance.py` omits it but adds `X-OpenBox-SDK-Version`. The User-Agent strings differ (`OpenBox-LangGraph-SDK` vs `OpenBox-SDK`). This will cause confusion in server-side request attribution.

**Recommendation:** Consolidate into a single source of truth (likely `client.py`) and import it in `hook_governance.py`. Reconcile the header differences.

---

## Finding 2: HIGH — Four parallel `_build_*_span_data()` functions with identical skeleton

All four functions share the same ~25-line structural pattern:

| File | Function | Lines |
|---|---|---|
| `http_governance_hooks.py:74` | `_build_http_span_data()` | 74-128 |
| `db_governance_hooks.py:88` | `_build_db_span_data()` | 88-156 |
| `file_governance_hooks.py:23` | `_build_file_span_data()` | 23-87 |
| `tracing.py:33` | `_build_traced_span_data()` | 33-74 |

**Identical skeleton across all four:**

1. Call `_hook_gov.extract_span_context(span)` to get `(span_id_hex, trace_id_hex, parent_span_id)`
2. Extract `raw_attrs` via `getattr(span, 'attributes', None)` then `dict(raw_attrs) if raw_attrs else {}`
3. Compute `now_ns = time.time_ns()`, `duration_ns`, `end_time`, `start_time` from the same formula
4. Return a dict with keys: `span_id`, `trace_id`, `parent_span_id`, `name`, `kind`, `stage`, `start_time`, `end_time`, `duration_ns`, `attributes`, `status`, `events`, `hook_type`
5. Then add type-specific fields at root level (http_url, db_statement, file_path, function)

**Recommendation:** Extract a `_build_base_span_data(span, name, kind, stage, hook_type, duration_ms=None, error=None) -> dict` helper in `hook_governance.py`. Each module's `_build_*_span_data()` calls it and extends with type-specific fields. Eliminates ~60 lines of near-identical timing/context code.

---

## Finding 3: HIGH — httpx request body extraction duplicated 3 times

The httpx request body extraction logic (try stream._stream, stream.body, stream._body, fallback to _content, then decode utf-8) appears in three places:

| Location | Lines | Context |
|---|---|---|
| `http_governance_hooks.py:256-287` | `_httpx_request_hook()` | Sync hook — full stream probing |
| `http_governance_hooks.py:376-403` | `_httpx_async_request_hook()` | Async hook — near-identical copy |
| `http_governance_hooks.py:487-508` | `_capture_httpx_request_data()` | Refactored helper — but only used in _patched_send |

The `_capture_httpx_request_data()` at line 487 was created to deduplicate this logic, but `_httpx_request_hook()` (line 256) and `_httpx_async_request_hook()` (line 376) still use hand-rolled inline versions instead of calling it.

**Recommendation:** Make `_httpx_request_hook` and `_httpx_async_request_hook` call `_capture_httpx_request_data()` instead of re-implementing the same stream-probing logic inline.

---

## Finding 4: HIGH — `_safe_serialize()` in tracing.py vs `safe_serialize()` in types.py

| Location | Function | Behavior |
|---|---|---|
| `types.py:458` | `safe_serialize(value)` | Recursive, handles LangChain model_dump/dict, no truncation |
| `tracing.py:88` | `_safe_serialize(value, max_length=2000)` | Flat, truncates to max_length, returns `"<unserializable>"` on error |

**Impact:** Two serialization functions with different names, different semantics, and different truncation behavior. `tracing.py` does not import or extend `types.safe_serialize`. A caller using `@traced(capture_result=True)` gets `_safe_serialize` truncation, while the same value in a governance event uses `safe_serialize` without truncation.

**Recommendation:** Extend `types.safe_serialize()` with an optional `max_length` parameter and use it everywhere. Or keep both but rename `_safe_serialize` to `_truncated_serialize` to make the distinction intentional and documented.

---

## Finding 5: MEDIUM — URL ignore check duplication (3 implementations)

URL-prefix ignoring is implemented in three separate places:

| Location | Function | Data Source |
|---|---|---|
| `http_governance_hooks.py:56` | `_should_ignore_url(url)` | Reads `_otel._ignored_url_prefixes` (module-level set in otel_setup.py) |
| `span_processor.py:53` | `WorkflowSpanProcessor._should_ignore_span(span)` | Reads `self._ignored_url_prefixes` (instance-level set) |
| `otel_setup.py:36` | Global `_ignored_url_prefixes` set | Written by `setup_opentelemetry_for_governance()` |

All three do the same loop: `for prefix in prefixes: if url.startswith(prefix): return True`. The ignored URL set is stored in two places (`otel_setup._ignored_url_prefixes` and `WorkflowSpanProcessor._ignored_url_prefixes`), which can diverge.

**Recommendation:** Keep one canonical set in `WorkflowSpanProcessor` and have `_should_ignore_url()` read from it via `hook_governance.get_span_processor()`.

---

## Finding 6: MEDIUM — `_otel._span_processor is None` guard duplicated 9 times

Every hook function in `http_governance_hooks.py` starts with:
```python
if _otel._span_processor is None:
    return
```

This appears at lines: 143, 183, 243, 319, 361, 423, 645, 694, 750.

Meanwhile, `hook_governance.py` has `is_configured()` which checks `_span_processor is not None` (line 92-94), and every hook already checks `_hook_gov.is_configured()` further down.

**Impact:** The `_otel._span_processor is None` check is redundant with the `is_configured()` check that follows. The early return prevents OTel span attribute capture (non-governance), but that's a separate concern from the governance check.

**Recommendation:** If the intent is "skip all instrumentation when OTel not set up", make this a single function. If it's intentionally separate from `is_configured()`, add a comment explaining why both checks are needed.

---

## Finding 7: MEDIUM — `import time as _time` inside function bodies (11 occurrences)

`time` is imported as `_time` inside function bodies in:
- `http_governance_hooks.py`: lines 90, 157, 198, 581, 603, 670, 723 (7 times)
- `file_governance_hooks.py`: line 41
- `tracing.py`: lines 42, 173, 229

Meanwhile, `db_governance_hooks.py` imports `time` at module level (line 22).

**Recommendation:** Import `time` at module level in all files. The lazy import provides no benefit here since `time` is a stdlib module with no circular import risk.

---

## Finding 8: MEDIUM — Timing pattern (perf_counter start/stop) duplicated everywhere

The pattern:
```python
start = time.perf_counter()
result = do_thing()
duration_ms = (time.perf_counter() - start) * 1000
```

appears 15+ times across db_governance_hooks.py, http_governance_hooks.py, and tracing.py. Each occurrence also includes identical `try/except` wrapping for governance blocked errors.

**Recommendation:** Not strictly a function to extract (context managers add overhead to hot paths), but a `@contextmanager` or utility like `measure_ms()` returning a callable could reduce ~3 lines per occurrence. Lower priority than other findings.

---

## Finding 9: MEDIUM — db_governance_hooks.py has sync/async near-identical copies

| Sync version | Async version | Shared code % |
|---|---|---|
| `_evaluate_db_sync()` (L164-185) | `_evaluate_db_async()` (L188-209) | ~95% |
| `_gov_traced_execution()` (L252-306) | `_gov_traced_execution_async()` (L308-358) | ~95% |

Both `_evaluate_db_sync/async` are thin wrappers that differ only in `_hook_gov.evaluate_sync` vs `await _hook_gov.evaluate_async`. The CursorTracer patches (`_gov_traced_execution` and `_gov_traced_execution_async`) are also near-identical, differing only in `await` keywords.

**Recommendation:** Consider a builder function that generates both sync/async variants from a shared template, or at minimum, extract the common span_data construction into a helper.

---

## Finding 10: LOW — GovernanceBlockedError import pattern inconsistency

`GovernanceBlockedError` is imported in different ways:
- `hook_governance.py`: `from openbox_langgraph.errors import GovernanceBlockedError` (inside function bodies, 4 times)
- `db_governance_hooks.py`: `from openbox_langgraph.errors import GovernanceBlockedError` (inside try/except blocks, 4 times)
- `file_governance_hooks.py`: `from openbox_langgraph.errors import GovernanceBlockedError` (inside function bodies, 4 times)
- `verdict_handler.py`: Module-level import

The lazy import is used to avoid circular imports, which is valid. However, the same lazy import appears 12+ times across files.

**Recommendation:** Consider making the import at module level where possible (db_governance_hooks.py and file_governance_hooks.py don't have circular deps with errors.py) or accept the pattern as necessary for circular import avoidance in hook_governance.py.

---

## Finding 11: LOW — `_httpx_response_hook` and `_httpx_async_response_hook` capture response but don't evaluate

Both `_httpx_response_hook` (line 307-357) and `_httpx_async_response_hook` (line 421-479) contain response body extraction logic followed by a comment:
```python
# NOTE: "completed" governance evaluation is handled in _patched_send
```

The response body/header extraction code in these hooks is dead code -- it captures `resp_body` and `resp_headers` into local variables that are never used.

**Recommendation:** Remove the unused response body capture code from these hook functions, or store it for later use if that's the intent.

---

## Summary Table

| # | Severity | Finding | Files Affected |
|---|---|---|---|
| 1 | CRITICAL | Duplicate `build_auth_headers()` with different outputs | client.py, hook_governance.py |
| 2 | HIGH | 4 parallel `_build_*_span_data()` with identical skeleton | http_governance_hooks, db_governance_hooks, file_governance_hooks, tracing |
| 3 | HIGH | httpx body extraction duplicated 3x | http_governance_hooks.py |
| 4 | HIGH | Two `safe_serialize` variants with different semantics | types.py, tracing.py |
| 5 | MEDIUM | URL ignore check in 3 places with 2 data sources | http_governance_hooks, span_processor, otel_setup |
| 6 | MEDIUM | `_span_processor is None` guard 9x, redundant with `is_configured()` | http_governance_hooks.py |
| 7 | MEDIUM | `import time as _time` lazy-imported 11 times unnecessarily | http_governance_hooks, file_governance_hooks, tracing |
| 8 | MEDIUM | perf_counter timing pattern 15+ times | db_governance_hooks, http_governance_hooks, tracing |
| 9 | MEDIUM | sync/async near-duplicate pairs in db_governance_hooks | db_governance_hooks.py |
| 10 | LOW | GovernanceBlockedError lazy-imported 12+ times | hook_governance, db_governance_hooks, file_governance_hooks |
| 11 | LOW | Dead response capture code in httpx response hooks | http_governance_hooks.py |

---

## Unresolved Questions

1. Is the header divergence in Finding 1 intentional (hook-level calls should have different User-Agent than SDK-level calls)?
2. For Finding 6, is the `_otel._span_processor is None` guard intended to gate non-governance span attribute writes, or is it purely a governance guard that's redundant with `is_configured()`?
3. For Finding 11, is the response capture in the httpx response hooks pre-work for a future feature, or truly dead code?
