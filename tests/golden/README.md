# Golden wire-payload baseline

Baseline oracle for the `openbox_langgraph` SDK's governance wire protocol —
what `GovernanceClient.evaluate_event` and `OpenBoxLangGraphHandler.ainvoke`
put on the wire and in what order, captured
against a mock HTTP transport (no real network, no simulated governance
logic). A future refactor of the base SDK wiring will regenerate this same
capture and byte-diff it against the fixtures committed here — that diff is
the actual regression gate. This README documents what's frozen, how it was
produced, and what is (and isn't) covered here vs. elsewhere in the suite.

## Pre-migration baseline status (measured 2026-07-02, `uv run --extra dev`)

| Gate | Result |
|------|--------|
| `pytest` | **80 passed** pre-migration → **119** with the baseline tests added here |
| `ruff check .` | **18 errors** — ALL pre-existing, only in `test_governance_changes.py` (16) + `test_telemetry_payload.py` (2). Package source is ruff-clean. |
| `mypy` (strict) | **135 errors in 9 source files** — ALL pre-existing, in the legacy hook/otel/tracing/span modules (db 37, http 31, file 24, span_processor 13, hook_governance 10, tracing 8, handler 7, otel_setup 5, verdict_handler 2). |

The migration must not INCREASE these: new/changed code adds ZERO ruff/mypy
errors and is itself strict-clean. Cleaning up the 135 pre-existing mypy errors
(legacy instrumentation modules, untouched until the follow-up hook-flip work)
is out of scope. `ruff`/`mypy`/`pytest` require `--extra dev`.

Branch-setup: base = `origin/main` (has DID signing + v0.2.0 the migration
builds on); modified `uv.lock` discarded (regenerable); untracked `demo/` left
local (the conformance kit is the authoritative parity gate); no worktree.

## What's frozen vs. what isn't

- **`*.normalized.json`** — the FROZEN expected values. Volatile fields
  (timestamps, nonces, signatures, generated ids) are replaced with stable
  placeholders (see "Normalization" below) so re-running the capture produces
  a byte-identical file. A future parity gate diffs against these, not the
  raw ones.
- **`*.raw.json`** — one real, un-normalized example of the same body, kept
  alongside for human readability (e.g. to see an actual timestamp or id
  shape). NOT diffed for parity — regenerating changes these by design.
- **`ordering_*.json`** — Layer 3 full-graph event orderings, normalized the
  same way as wire bodies (both id and event_type fields are stable).

## What this baseline does NOT cover (covered elsewhere)

Live request **signing correctness** — that `X-OpenBox-Body-SHA256` actually
equals `sha256(the exact bytes sent)`, and that the Ed25519 signature
verifies against the canonical request string — is covered independently by
`tests/test_did_client_signing.py`, which asserts those relationships
functionally against a live mock transport rather than golden-matching a
frozen placeholder value. This baseline's `headers_signed.normalized.json`
only pins WHICH header names are present when a client is signed vs.
unsigned; it does not (and should not) try to re-verify the signature math,
since the signature value is normalized away as volatile.

## Files

| File | Role |
|------|------|
| `id_normalization.py` | `normalize_json()` — the single source of truth for volatile→placeholder substitution. |
| `capture_harness.py` | Layer 1/2 primitives: recording mock transport, `write_fixture_pair`/`write_ordering_fixture`, `capture_single_event`, test-only DID/key constants. |
| `fake_agent_graphs.py` | Minimal compiled LangGraph `StateGraph`s driven by `FakeMessagesListChatModel` (no network LLM call). |
| `graph_capture_harness.py` | Layer 3: `OrderedCapture` (records both event ordering AND full wire bodies), `RecordingGovernanceClient` (injection seam, supports verdict overrides), `run_captured_ainvoke`/`run_ordered_capture`. |
| `real_emitted_event_fixtures.py` | Drives real `handler.ainvoke()` runs (baseline / tool / subagent / error scenarios) and extracts the real wire body for each lifecycle event type. |
| `layer1_handbuilt_pins.py` | The two verified-infeasible event types (see below) — hand-built, clearly labelled. |
| `generate_fixtures.py` | Orchestrates all of the above. Run: `uv run --extra dev python3 -m tests.golden.generate_fixtures`. |

Consumed by `tests/test_golden_baseline.py` (existence/non-emptiness,
structural-key checks on real-emitted bodies, signed vs. unsigned header sets,
two-LLM-call de-dup guard).

> Note: legacy in-repo hook governance was removed — hook payloads/spans are
> now owned by the base `openbox_core` instrumentation. The old
> `hook_trigger.*` fixtures + `hook_trigger_fixture.py` capture were retired
> with it; this oracle now covers lifecycle wire bodies only.

## Real-emitted vs. hand-built

Every Layer 1 wire body is captured from a REAL `OpenBoxLangGraphHandler.ainvoke()`
run — via `RecordingGovernanceClient` injected as the handler's `client=` —
**except** two event types that are genuinely unreachable by any real code
path (verified, not assumed):

- **`workflow_failed.handbuilt_serialization_pin.*`** — grep across
  `openbox_langgraph/*.py` finds zero construction sites for
  `event_type="WorkflowFailed"` (or its SDK-internal source label,
  `"ChainFailed"`). It exists only as an enum value and a
  `to_server_event_type` mapping target; nothing in the handler ever builds
  one today.
- **`chain_started.handbuilt_serialization_pin.*`** — a root `ChainStarted`
  from `on_chain_start` is real dead code: `_process_event` only reaches the
  send when `send_chain_start_event=False`, but that exact same flag also
  gates `_pre_screen_input`'s `WorkflowStarted` send, and `_process_event`'s
  own skip-guard for `ChainStarted` ALSO fires when that flag is `False`.
  The two guards are mutually exclusive on the same config value — no
  configuration reaches the code that would send it. Its wire `event_type`
  is `"WorkflowStarted"` (server-mapped from the SDK-internal
  `"ChainStarted"` label), not the literal string `"ChainStarted"`.

Both are written via `write_fixture_pair(..., handbuilt_pin=True)`, which
adds the `.handbuilt_serialization_pin` marker to both filenames so it's
unambiguous — including to a future parity gate — that these pin a
hand-authored event's client-side serialization only, not a real handler
emission. If a future change adds a real construction site for either, these
should be replaced by a real-emitted capture and this note updated.

## Capture method

Reuses the proven `httpx.MockTransport` seam from `test_did_client_signing.py`:

- **Lifecycle bodies:** `client._client = httpx.AsyncClient(transport=MockTransport(rec))`.
- **Full-graph capture (real handler.ainvoke):** a real compiled `StateGraph` +
  fake chat model, wrapped by `OpenBoxLangGraphHandler` with an injected
  `RecordingGovernanceClient` (`client=` option) — no mock transport needed
  here since the client itself is the seam; `evaluate_event` records the
  event before returning a verdict (default ALLOW, or an override to force a
  specific verdict, e.g. HALT to trigger the error-close scenario).

Test keys are the repo's existing non-real constants (`bytes(range(32))`
seed; `did:aip:550e8400-…`) — no real keys committed.

## Normalization

`id_normalization.normalize_json()` replaces, by VALUE SHAPE (not key name,
with one exception):

- RFC3339 timestamps → `<TS>`.
- A bare canonical UUID, or the handler's own `{prefix}-(run-)?{hex8}`
  generated-id scheme, each with an optional literal suffix
  (`-wf`/`-pre`/`-sig`/`-c`/`-pre-c`) preserved on the placeholder — e.g.
  `<ID>-pre-c` — so which row an activity_id refers to stays visible. Also
  handles a UUID embedded as a substring inside a longer string (e.g.
  LangChain's own `AIMessage.id`, shaped `lc_run--{uuid7}-0`).
- Ed25519 signatures / SHA256 body digests → `<NONCE>`.
- `duration_ms` / `start_time` / `end_time` → `<DURATION_MS>`. The one
  exception normalized by KEY NAME rather than value shape — a real elapsed-
  time float has no shape signal distinguishing it from any other float.

Hand-authored literal ids in the pin fixtures (e.g. `chain-run-1`) are NOT
normalized — the generated-id pattern specifically requires 8 hex characters,
which decimal-suffixed literals never match.

**Reproducibility is verified, not assumed:** regenerating twice in a row
(`uv run --extra dev python3 -m tests.golden.generate_fixtures` run twice)
produces a byte-identical `git diff --stat` against `*.normalized.json` and
`ordering_*.json` — empty output. `.raw.json` is intentionally excluded from
that check; it is not normalized and will differ between runs.

## Wire-shape facts worth knowing before diffing against this baseline

- `LangChainGovernanceEvent.to_dict()` filters ONLY `None` — falsy defaults
  like `hook_trigger: false` DO appear on the wire. A naive "omit falsy"
  normalizer would silently break parity.
- LLMCompleted's wire `event_type` is `ActivityCompleted`, activity_id
  `…-pre-c` (or `…-c` for a non-pre-screened LLM call) — it never carries a
  `spans` key (no hook fires for LLM calls).
- A healthy run's root-close event is SDK-internal `ChainCompleted`, wire
  `event_type` `WorkflowCompleted` (via `to_server_event_type`). The
  LITERAL SDK-internal `WorkflowCompleted` only exists on the error-close
  path (`workflow_completed_error_close.*`, `status: "failed"`) — the two
  are genuinely different events, not duplicates.
- A two-LLM-call turn produces two fully independent `LLMStarted`/
  `LLMCompleted` pairs with distinct activity_ids — so they do NOT collide
  in the de-dup store (which keys on `(activity_id, event_type)`). That
  store lives in `GovernanceClient` (`client.py` `_dedup_run`/`_dedup_sent`
  + `_is_duplicate`), scoped per `(workflow_id, run_id)` and reset each run;
  it is real, not absent. The handler additionally skips re-sending the
  pre-screen `LLMStarted` from `_process_event` — a separate avoidance.
- Subagent-labelled Tool events (`resolve_subagent_name` set) carry
  `subagent_name` and `tool_type: "a2a"` on the wire, plus an appended
  `{"__openbox": {...}}` entry in `activity_input` for Rego policy use.
- Signed vs. unsigned client: signed emits all five `X-OpenBox-Agent-*` /
  `X-OpenBox-Body-SHA256` headers; unsigned emits none.
