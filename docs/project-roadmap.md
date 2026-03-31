# OpenBox LangGraph SDK — Project Roadmap

**Last Updated:** 2026-03-21 | **Version:** 0.1.0 (Beta) | **Status:** Active Development

## Current Phase: Consolidation & Bug Fixes (P1)

**Duration:** 2026-03-21 to 2026-04-04 (2 weeks)
**Focus:** Stabilize core governance, port proven patterns from DeepAgent SDK

### Active Work (In Progress)

#### 1. Remove SpanCollector (Branch: feat/otel-http-hook-spans)
**Status:** In progress | **Priority:** P1 | **Effort:** 2 days

Simplify activity context to use SpanProcessor only (no dual ContextVar registry).

**Plan:** `plans/260320-0329-remove-span-collector/plan.md`

**Changes:**
- Remove SpanCollector imports from langgraph_handler.py
- Remove SpanCollector from hook_governance.py
- Delete span_collector.py
- Cleanup __init__.py exports

**Risk:** Internal span hooks (started/completed) will be gone — Core won't see function_call hook_type spans.

**Resolution:** Low priority; hooks now rely on built-in instrumentation.

#### 2. Port DeepAgent Fixes (Branch: feat/otel-http-hook-spans)
**Status:** Planned | **Priority:** P1 | **Effort:** 1 day

Merge proven patterns from openbox-deepagent SDK.

**Plan:** `plans/260321-2019-port-deepagent-fixes/`

**Phase 1 — Remove HITL Gates:**
- Remove `hitl.enabled` check (always poll if REQUIRE_APPROVAL)
- Remove `hitl.skip_tool_types` nesting (use GovernanceConfig.skip_tool_types instead)
- Simplifies HITL call sites in langgraph_handler.py

**Phase 2 — Add sqlalchemy_engine Parameter:**
- Expose `sqlalchemy_engine` kwarg in create_openbox_graph_handler()
- Thread to setup_opentelemetry_for_governance()
- Fixes DB governance not firing when engine created before handler

**Phase 3 — Hook-Level HITL Retry:**
- If HTTP/DB hook returns REQUIRE_APPROVAL, poll HITL at hook level
- Currently only LLM/chain/tool events trigger HITL
- Enables "require approval before reading sensitive DB table" policies

---

## Planned Phases (Q2 2026)

### Phase 2: Enhanced Observability & Metrics (2026-04-04 to 2026-04-18)
**Priority:** P2 | **Effort:** 3 weeks

**Scope:**
1. **Verdict Metrics** — Track verdict distribution (allow, block, halt, approval) per tool/LLM
2. **Latency Tracking** — Measure governance evaluation latency (Core API, hook evaluation)
3. **HITL Metrics** — Approval rate, average wait time, rejection rate
4. **Error Tracking** — API errors, network timeouts, dedup stats

**Deliverables:**
- Prometheus-compatible metrics export
- Metrics dashboard in OpenBox Core
- Built-in alerting for high block rate or evaluation latency

**Risk:** Overhead of metric collection in hot paths (hooks)
**Mitigation:** Sampling, async metric aggregation

---

### Phase 3: Advanced Hook Governance (2026-04-18 to 2026-05-02)
**Priority:** P2 | **Effort:** 3 weeks

**Scope:**
1. **Response Body Governance** — Full HTTP response body capture and evaluation
   - Currently: response body in completed stage only (informational)
   - Future: allow policies to inspect response and decide to redact/truncate
2. **Streaming Response Handling** — Chunked HTTP responses, Server-Sent Events
   - Current httpx hook sees only fully-buffered responses
   - Future: support streaming (chunk-by-chunk evaluation)
3. **Database Result Governance** — Row-level access control
   - Currently: query blocked before execution
   - Future: allow query, but filter/redact rows in result set

**Deliverables:**
- Hook payload schema v2 (extended body capture)
- Rego policy examples for response redaction
- Performance benchmarks (streaming overhead)

**Risk:** Complexity spike; need careful error handling for partial results
**Mitigation:** Phase as: v1 (response capture), v1.5 (redaction), v2 (streaming)

---

### Phase 4: HITL & Approval UX (2026-05-02 to 2026-05-16)
**Priority:** P2 | **Effort:** 2 weeks

**Scope:**
1. **Approval Context** — Include rich context in approval request
   - Tool name, input, LLM prompt, reasoning
   - Allow approvers to see what they're approving
2. **Contextual Approval** — Reusable approvals
   - "Always allow search_web for the next 5 minutes"
   - "Allow this specific query on this dataset"
3. **Approval Delegation** — Route to specific teams/roles
   - "Send to #security-team if risk_score > 0.8"
   - "Require 2/3 approvals for financial transactions"

**Deliverables:**
- Extended ApprovalResponse schema (context fields)
- Delegation rules in GovernanceConfig
- Approval deduplication (avoid prompting twice for same action)

**Risk:** Scope creep; delegation logic in Core vs SDK
**Mitigation:** MVP: v1 (context only), v1.5 (delegation in Core)

---

### Phase 5: Subagent & Multi-Agent Orchestration (2026-05-16 to 2026-06-13)
**Priority:** P3 | **Effort:** 4 weeks

**Scope:**
1. **Nested Agent Tracking** — Identify subagent calls in span hierarchy
   - Currently: tool_type="a2a" marks subagent
   - Future: track full call stack (root agent → subagent-1 → subagent-2 → tool)
2. **Cross-Agent Authorization** — Policies aware of call chain
   - "Subagent can only call database tools, not external APIs"
   - "Service-level API keys per subagent"
3. **Agent-to-Agent Token Isolation** — No token leakage between agents
   - Each subagent gets scoped API key
   - Governance enforces API key scope

**Deliverables:**
- Span hierarchy tracking in WorkflowSpanProcessor
- Agent identity resolution in hook governance
- Scope-based API key enforcement

**Risk:** Deep integration with LangGraph's agent execution model
**Mitigation:** Start with single-level (root + direct children), expand later

---

## Long-Term Vision (2026 Q3+)

### Plugin Architecture for Custom Hooks
- Extensible hook registration: `register_custom_hook(lib_name, intercept_fn)`
- Community hooks: FastAPI, gRPC, cloud SDKs (AWS, GCP)

### Governance as Code (Rego Auto-Generation)
- SDK-level policy builder: `policy = Policy(name="...", rules=[...])` → generates Rego
- TypeScript parity: API mirrors @openbox/sdk-langgraph

### Real-Time Policy Enforcement Feedback
- Agent sees why it was blocked/constrained
- Automatic prompt adjustment (e.g., "try a different search query")

### Deployment Patterns
- Local governance (offline mode, cached policies)
- Edge deployment (VPC, airgap)
- Multi-region failover

---

## Feature Requests & Community Backlog

### High-Priority Community Requests
1. **Sync-Only Applications** — Async-free alternative for sync agents
   - Currently: sync wrappers exist but not fully documented
   - Request: built-in sync handler class
2. **LangChain Tool Compatibility** — Direct integration with LangChain tools
   - Currently: must use LangGraph (no direct tool governance)
   - Request: standalone tool wrapper
3. **Guardrails V2 Integration** — Use Guardrails SDK's validators
   - Currently: homegrown PII/toxicity
   - Future: plug in Guardrails validators via Core

### Medium-Priority Requests
- Batch inference governance (await batch(...) with governance)
- Multi-turn conversation governance (context-aware verdict per message)
- Cost tracking (track token usage per verdict)
- Compliance audit trails (immutable logs for SOC 2 / HIPAA)

---

## Deprecation Timeline

### v0.1.0 (Current — Beta)
- No breaking changes expected
- API surface stable but documented as beta

### v0.2.0 (Expected: 2026-04-30)
- Stabilize event schema (may rename fields for clarity)
- Deprecate pre-0.1.0 custom hooks (if any)

### v1.0.0 (Expected: 2026-06-30)
- Semantic versioning kicks in
- API guaranteed stable until v2.0.0
- 12-month deprecation notice before breaking changes

---

## Metrics & Success Criteria

### Developer Adoption
- **Goal:** 50+ production deployments by end of Q2 2026
- **Metric:** GitHub stars, npm downloads, community contributions

### Reliability
- **Goal:** <0.1% failure rate in fail_open mode
- **Metric:** Monitoring dashboard in OpenBox Core
- **SLA:** 99.9% availability of OpenBox API for non-localhost URLs

### Performance
- **Goal:** <50ms latency for hook evaluation (50th percentile)
- **Metric:** Built-in performance dashboard
- **Budget:** Hook evaluation + network latency to Core ≤ 50ms

### Community Health
- **Goal:** Response time <24h for GitHub issues
- **Metric:** Issue triage automation
- **Goal:** Weekly updates in changelog
- **Metric:** Release cadence tracking

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| OpenBox Core API unreachable | Medium | High | fail_open mode, cache policies |
| Performance regression in LangGraph | Low | Medium | Benchmarking in CI/CD |
| Breaking change in LangGraph v0.3 | Medium | High | Version pinning, early testing |
| Sync context issues in edge cases | Medium | Medium | Extensive async/sync testing |
| Hook order dependency (HTTP before DB) | Low | Medium | Explicit hook registration order |
| Memory leak in ContextVar cleanup | Low | High | Memory profiling, tests |

---

## Backlog (Unscheduled)

### Documentation
- [ ] API reference docs (auto-generated from docstrings)
- [ ] Tutorial: 5-minute governance setup
- [ ] Troubleshooting guide (common issues & fixes)
- [ ] Rego policy cookbook (10+ policy examples)

### Testing
- [ ] Load tests (1000 req/sec governance evaluation)
- [ ] Chaos testing (Core API intermittently unreachable)
- [ ] Interop tests with popular LLM SDKs (OpenAI, Anthropic, Claude)

### Infrastructure
- [ ] Docker image with SDK pre-installed
- [ ] GitHub Actions integration (auto-test on PR)
- [ ] PyPI automated releases

### Code Quality
- [ ] Code coverage to 90%+
- [ ] Dependency audit (security)
- [ ] Type stub generation (.pyi files)

---

## Links & Resources

- **OpenBox Core:** https://dashboard.openbox.ai
- **OpenBox Docs:** https://docs.openbox.ai
- **LangGraph Docs:** https://langchain-ai.github.io/langgraph/
- **OPA/Rego:** https://www.openpolicyagent.org/
- **SDK GitHub:** https://github.com/openbox-ai/openbox-langgraph-sdk-python

---

## Contributors & Ownership

| Component | Owner | Contributors |
|-----------|-------|---------------|
| Event stream (langgraph_handler.py) | @tino | — |
| Hook governance | @tino | — |
| HTTP hooks | @tino | — |
| Database hooks | @tino | — |
| Instrumentation | @tino | — |
| Docs | @tino | — |

**How to Contribute:**
1. Read CONTRIBUTING.md (to be created)
2. Open an issue for discussion
3. Submit PR against feat/* branch
4. Ensure tests pass + mypy strict compliance
5. Update docs/CHANGELOG for user-facing changes

---

## FAQ

### Q: When is v1.0.0 stable release?
**A:** Expected 2026-06-30. v0.1.0 is production-ready but API may change.

### Q: Can I use this with FastAPI / Flask / Django?
**A:** No. SDK wraps LangGraph graphs specifically. For web frameworks, wrap the handler in your route.

### Q: What if OpenBox Core is down?
**A:** fail_open mode (default) lets execution continue; fail_closed mode blocks. Your choice.

### Q: Do I need to change my LangGraph code?
**A:** No. Wrap the compiled graph with `create_openbox_graph_handler()`, that's it.

### Q: How does this compare to LangSmith monitoring?
**A:** Complementary. LangSmith is observability + tracing; OpenBox is active governance + policy enforcement.

---

## Changelog

### v0.1.0 (2026-03-21)
- Initial beta release
- Event stream governance (pre-screen + inline)
- HTTP/DB/file hook governance
- Span tracking & activity context
- HITL approval queue
- Full type checking (mypy strict)

### v0.0.x (Pre-Release)
- Internal testing & validation
