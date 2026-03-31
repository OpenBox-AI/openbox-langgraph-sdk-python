# Efficiency Review: feat/otel-http-hook-spans

**Date:** 2026-03-23
**Branch:** feat/otel-http-hook-spans
**Reviewer:** code-reviewer agent (Opus 4.6)

---

## Scope

- **Files reviewed:** 12 source files (~4,600 LOC)
- **Focus:** Efficiency — redundant work, missed concurrency, hot-path bloat, memory, no-op updates

## Overall Assessment

The hook-level governance adds a synchronous HTTP call to OpenBox Core on **every** HTTP request, DB query, file I/O operation, and `@traced` function call. This is by design (real-time governance), but several implementation patterns amplify the per-operation overhead beyond what's necessary.

---

## Critical Issues

### C1. Double JSON serialization on every hook-governance call
**File:** `hook_governance.py` lines 188-191
**Impact:** Every governance evaluation pays for `json.dumps(payload)` purely as a validation check, then httpx serializes the same payload again internally. The fallback at line 191 does `json.loads(json.dumps(payload, default=str))` — a third serialization pass.
```python
try:
    json.dumps(payload)           # <-- validation-only, result discarded
except (TypeError, ValueError):
    payload = json.loads(json.dumps(payload, default=str))  # <-- double-encode fallback
```
**Recommendation:** Remove the validation pass. Use a single `json.dumps(payload, default=str)` only when `httpx.post(json=...)` raises, or serialize once and pass `content=` + `headers=` to httpx.

### C2. Auth headers rebuilt on every GovernanceClient call
**File:** `client.py` line 329
**Impact:** `_headers()` calls `build_auth_headers(self._api_key)` (allocates a new dict) on every `evaluate_event`, `poll_approval`, `evaluate_raw`, and `validate_api_key` call. These are static after construction.
```python
def _headers(self) -> dict[str, str]:
    return build_auth_headers(self._api_key)  # new dict allocation every call
```
**Recommendation:** Cache headers at `__init__` time (like `hook_governance.py` already does with `_cached_auth_headers`).

### C3. Sync `evaluate_event_sync` creates a new `httpx.Client` per call
**File:** `client.py` lines 210-216
**Impact:** Each sync governance call opens a fresh TCP connection (no connection reuse). This adds TLS handshake + TCP setup latency to every `invoke()` / `stream()` call.
```python
with httpx.Client(timeout=self._timeout) as client:
    response = client.post(...)
```
**Recommendation:** Use a persistent `httpx.Client` (lazy-init like `_get_client()` does for async).

---

## High Priority

### H1. Redundant `import time as _time` inside hot-path functions
**Files:** `http_governance_hooks.py` (7 occurrences: lines 90, 157, 198, 581, 603, 670, 723), `file_governance_hooks.py` (line 41), `tracing.py` (lines 42, 173, 229)
**Impact:** While CPython caches module lookups after first import, the `import` statement still executes bytecode on every call (IMPORT_NAME + STORE_FAST). On the HTTP hot path this runs 7 times per request cycle.
**Recommendation:** Move `import time` to module level. There is no circular-import concern with stdlib `time`.

### H2. `_should_ignore_url()` iterates prefix set on every HTTP operation
**File:** `http_governance_hooks.py` lines 56-63
**Impact:** Called at least twice per HTTP request (request hook + response hook), plus once in `_patched_send`. Each call iterates `_otel._ignored_url_prefixes` linearly via `str.startswith`.
**Recommendation:** If the set is small this is fine. If it grows, consider a trie or at minimum converting to a tuple for faster iteration than set.

### H3. `_build_payload` does `span.get_span_context()` twice per evaluation
**File:** `hook_governance.py` lines 162 and then `extract_span_context` (line 108) called from the hook module's `_build_*_span_data` function which runs **before** `_build_payload` is called.
**Impact:** The span context is extracted twice per governance call: once in the caller's `_build_http_span_data` / `_build_db_span_data` / etc. (to populate `span_id`, `trace_id`), then again in `_build_payload` (to look up `activity_context`). These are the same span.
**Recommendation:** Pass `trace_id` from the already-extracted span context into `_build_payload` to avoid the redundant `get_span_context()` call.

### H4. Request body extracted multiple times per httpx request
**File:** `http_governance_hooks.py`
**Impact:** For httpx requests, the body is extracted in:
1. `_httpx_request_hook` (lines 251-287) — for "started" governance
2. `_capture_httpx_request_data` called from `_patched_send` (lines 586) — for "completed" governance

The same `request._content` / `request.stream._stream` attribute walking happens twice per request. Similarly for the response body in `_httpx_response_hook` + `_capture_httpx_response_data`.
**Recommendation:** Store the extracted body in the ContextVar alongside the span, or skip body extraction in the response hook since `_patched_send` handles completed governance.

### H5. `_httpx_response_hook` / `_httpx_async_response_hook` do work but never send governance
**File:** `http_governance_hooks.py` lines 307-357 (sync), 421-479 (async)
**Impact:** Both response hooks extract headers and body but end with a comment: `# NOTE: "completed" governance evaluation is handled in _patched_send`. The work is pure waste — headers are re-extracted, body is re-extracted, then discarded.
**Recommendation:** Remove body/header extraction from response hooks since `_patched_send` handles everything. Keep only the early-return guards.

### H6. File I/O: governance evaluation fires twice per read/write operation
**File:** `file_governance_hooks.py` lines 162-175 (read), 212-225 (write)
**Impact:** Every `TracedFile.read()`, `write()`, `readline()`, `readlines()`, `writelines()` sends both a "started" AND "completed" governance HTTP call. That's 2 synchronous HTTP round-trips per file operation. For a file that does 10 reads, that's 20 governance API calls plus 2 more for open/close.
**Recommendation:** Consider batching or making "completed" stage fire-and-forget (async). At minimum, make the "completed" evaluation non-blocking for read/write.

### H7. `TracedFile` governance on `close()` always fires even when no operations were performed
**File:** `file_governance_hooks.py` lines 247-268
**Impact:** Even if the file is opened and immediately closed (common in existence checks), a governance HTTP call fires.
**Recommendation:** Guard with `if self._operations:` before sending close governance.

---

## Medium Priority

### M1. Module-level dict `.clear()` loses all entries when cap is hit (not LRU)
**Files:**
- `http_governance_hooks.py` line 163 (`_http_hook_timings`)
- `db_governance_hooks.py` line 533 (`_pymongo_pending_commands`), line 720 (`_redis_span_meta`), line 778 (`_sa_timings`)

**Impact:** When these dicts hit their cap (1000 entries), `.clear()` drops ALL entries including still-in-flight operations. A response hook looking up its request's start time will find nothing, returning incorrect duration (None or 0).
**Recommendation:** Use an LRU eviction strategy (e.g., `collections.OrderedDict` with popitem, or just pop the oldest N entries) instead of clearing everything.

### M2. `_http_hook_timings` is a global dict without thread safety
**File:** `http_governance_hooks.py` line 38
**Impact:** Multiple threads (e.g., from `requests` library in thread pool) can read/write `_http_hook_timings` concurrently. `.clear()` during iteration or concurrent `.pop()` could cause `RuntimeError: dictionary changed size during iteration` in CPython (though GIL makes this unlikely for simple ops, `.clear()` while another thread iterates is still unsafe).
**Recommendation:** Use a `threading.Lock` or switch to a thread-safe structure. Same applies to `_redis_span_meta`, `_sa_timings`, `_pymongo_pending_commands`.

### M3. `span_processor.unregister_workflow()` iterates all dicts on every cleanup
**File:** `span_processor.py` lines 90-102
**Impact:** `unregister_workflow` iterates `_aborted_activities`, `_halt_requests`, `_activity_context`, and `_trace_to_workflow` to find keys matching the workflow_id prefix. Under the lock. If many workflows have been registered, this is O(n) per cleanup.
**Recommendation:** Consider a nested dict structure (`workflow_id -> {activity_id -> data}`) for O(1) cleanup.

### M4. `rfc3339_now()` called redundantly in `_build_payload`
**File:** `hook_governance.py` line 185
**Impact:** `rfc3339_now()` is already called when building span_data in each hook module's `_build_*_span_data` function (which sets `start_time` via `time.time_ns()`). Then `_build_payload` calls it again for `payload["timestamp"]`. Minor, but it's an extra datetime allocation per governance call.

### M5. `_requests_response_hook` re-extracts request body already captured by request hook
**File:** `http_governance_hooks.py` lines 207-215
**Impact:** The request body is extracted again in the response hook for the "completed" governance call, even though it was already extracted in `_requests_request_hook`. Unlike httpx (which has `_patched_send`), the requests library hooks don't share data between request/response hooks.
**Recommendation:** Store request body in a dict keyed by span_id (same pattern as `_http_hook_timings`) to avoid re-extraction.

### M6. `_urllib3_response_hook` loses URL path information
**File:** `http_governance_hooks.py` lines 714-720
**Impact:** The response hook reconstructs URL as just `scheme://host:port/` (hardcoded `/`), losing the actual request path. The request_info is not passed to response_hook, so the URL in the "completed" governance event is less specific than "started".
**Recommendation:** Store the full URL from the request hook alongside timing data.

### M7. Duplicate `_resolve_tool_type` calls in chain start/end
**File:** `langgraph_handler.py` lines 1087+1143
**Impact:** For subagent chain events, `_resolve_tool_type` is called both on `on_chain_start` and `on_chain_end` with the same arguments. Minor since it's a dict lookup.

---

## Low Priority

### L1. `os.environ.get("OPENBOX_DEBUG")` checked on every API call
**File:** `client.py` lines 149, 158, 206, 288, 301
**Impact:** 2-5 env var lookups per governance event. `os.environ.get()` acquires a lock internally.
**Recommendation:** Cache at module init or instance init.

### L2. `_is_async_function` imports `asyncio` on every `@traced` decoration
**File:** `tracing.py` line 274
**Impact:** Only happens once per decorated function (at import time), so negligible.

### L3. `GovernanceVerdictResponse` used but not imported in `langgraph_handler.py`
**File:** `langgraph_handler.py` lines 99, 448
**Impact:** This is a type annotation issue (runtime it works due to `__future__.annotations`), but it's not an efficiency issue. Noting for completeness.

---

## Positive Observations

1. **Connection pooling in hook_governance.py** — Persistent `httpx.Client` / `AsyncClient` instances avoid per-call TCP overhead (unlike `client.py` sync path).
2. **Activity abort short-circuit** — `_check_activity_abort()` in `evaluate_sync/async` prevents sending governance calls for already-aborted activities.
3. **Deduplication in GovernanceClient** — `_is_duplicate()` prevents sending redundant `(activity_id, event_type)` pairs within a run.
4. **Capped timing dicts** — All timing/metadata dicts have caps (1000) preventing truly unbounded growth (though eviction strategy is suboptimal).
5. **`_skipPatterns` in file hooks** — System paths like `/dev/`, `/proc/`, `__pycache__` are filtered early.
6. **Workflow cleanup** — `unregister_workflow()` removes all associated state, preventing memory leaks across runs.

---

## Metrics

| Metric | Value |
|--------|-------|
| Governance HTTP calls per tool invocation | 2 (started + completed) + N per HTTP/DB/file op within |
| Governance HTTP calls per file read() | 2 (started + completed) |
| JSON serialization passes per hook evaluation | 2-3 (validation + httpx + possible fallback) |
| Auth header allocations per API call | 1 (client.py) or 0 (hook_governance.py cached) |
| `import time` bytecodes per HTTP request | 7+ |

---

## Recommended Actions (Priority Order)

1. **C1** — Remove validation-only `json.dumps` in `_build_payload`
2. **C3** — Use persistent `httpx.Client` in `evaluate_event_sync`
3. **C2** — Cache `_headers()` in `GovernanceClient.__init__`
4. **H5** — Remove dead work in `_httpx_response_hook` / `_httpx_async_response_hook`
5. **H4** — Avoid double body extraction for httpx requests
6. **H6** — Consider making "completed" file governance fire-and-forget
7. **H1** — Move `import time` to module level
8. **M1** — Replace `.clear()` with LRU eviction for timing dicts
9. **M2** — Add thread safety to global timing dicts
10. **M3** — Restructure span_processor dicts for O(1) workflow cleanup

## Unresolved Questions

1. Is the double-evaluation (started + completed) per file read/write intentional? The "completed" stage for reads carries the data content — this could be very large for big files and adds governance latency to every read call.
2. Should "completed" governance evaluations be fire-and-forget (async, non-blocking) since the operation has already happened? Currently they block the calling thread.
3. The `_patched_send` approach for httpx body capture duplicates work with the OTel response hooks. Is there a plan to consolidate, or are both paths needed for edge cases?
