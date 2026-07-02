# Changelog

## 0.3.0

Migrates the SDK onto the shared `openbox-sdk-python` (`openbox_core`) governance
core. The public API (classes, functions, `initialize()` signature, DID signing)
is preserved; the changes below are internal rewiring plus a small, deliberate
set of wire and semantic changes, each regression-gated by a reproducible
golden-payload oracle.

### Changed — wire protocol (lifecycle events)

- Lifecycle events (`SignalReceived`, `WorkflowStarted`/`Completed`,
  `LLMStarted`/`Completed`, `ToolStarted`/`Completed`) now serialize through the
  shared `openbox_core` `EventEnvelope` + strict gate instead of hand-built
  request bodies. The wire body is byte-identical to before **except** three
  compatibility-noise fields the base envelope omits by construction and no
  longer sends for lifecycle events: `hook_trigger: false`, `spans: []`,
  `span_count: 0`. Core treats an absent `hook_trigger` as `false`; this is the
  same envelope format the Temporal SDK already ships to Core.
- Lifecycle-event requests now carry the base SDK's `User-Agent`; hook-triggered
  requests keep `OpenBox-LangGraph-SDK/<version>`.

### Changed — semantics

- **Approval parsing is action-first and strict.** When both are present the
  `action` field now wins over `verdict` (was verdict-first). An unknown or
  empty approval response resolves to *pending* (keeps polling) — never an
  implicit approve. The previous lenient "unknown → allow" fallback at the
  human-approval boundary is removed.
- **Hook-level HITL approval retry now runs governed.** After an approval is
  granted, the pending abort state is cleared on both the base and legacy stores
  before the graph re-invokes. Previously the stale abort flag re-blocked the
  approved retry (`clear_activity_abort` had no caller) — a genuine bug, now
  fixed.
- **Fail-open is precise.** A governance transport/parse failure surfaces as
  "no verdict" (fail-open, unchanged) only for a genuine client-synthesized
  fallback; a real blocking verdict that happens to carry `fallback_used: true`
  still enforces.
- Expired-approval-before-allow ordering is unchanged (an expired approval
  raises even when allow-shaped).

### Added

- **`use_core_instrumentation`** config flag (default `False`). When enabled,
  covered hook families (`http`, `dbapi`, `asyncpg`, `sqlalchemy`, `file`) are
  governed through the shared `openbox_core` hook runtime while uncovered
  families (`urllib`, `urllib3`, `redis`, `mongo`) stay on the legacy hooks —
  each operation governed exactly once. Approval on this path is raise-only and
  never blocks the event loop. Per-family flips (making the core runtime the
  default for a family) remain follow-up work.
- `openbox_langgraph.__version__`.
- `multi_agent_session_id` config/handler option (distinct from `session_id`).
- Dependency on `openbox-sdk-python` (the shared governance core).

### Preserved (regression-gated)

DID request signing (byte-identical `X-OpenBox-Agent-*` headers), the
`(activity_id, event_type)` de-dup, transient-failure PII redaction, the
public `client=` injection seam, and every public class/function signature.
