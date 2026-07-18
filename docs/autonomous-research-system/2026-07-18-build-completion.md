# Research Desk build completion — 2026-07-18

## Outcome

Research Desk is a complete, locally runnable autonomous multi-agent research
system. A LangGraph supervisor coordinates Researcher, Summarizer, Critic, and
Report Writer workers through explicit, bounded routes. The same engine powers a
Streamlit workbench, FastAPI service, Typer CLI, Markdown/JSON/HTML exports,
SQLite checkpoints, a durable run ledger, and SQLite-backed vector recall.

The default workflow is keyless and deterministic. Live mode adds Tavily,
arXiv, uploaded PDF ingestion, and an OpenAI-compatible structured model without
silently substituting demo fixtures.

## Completion accounting

- Planned Objective Count: 9
- Completed Objective Count: 9
- Additional Objectives Completed: 8
- Unresolved Objective Count: 0
- Remaining Objective Count: 0
- Terminal State: Complete
- Extra Mile: Removed an unpatched vector-server dependency; added process-wide
  single-writer lifecycle, provider pacing, aggregate upload limits, untrusted-text
  rendering boundaries, default network isolation, independent wheel proof, and
  executable desktop/mobile plus container persistence proof.

The eight additional objectives were discovered during security, product,
current-truth, browser, and packaging audits. They are covered by automated
regressions rather than recorded as review-only recommendations.

## Architecture delivered

1. `ResearchEngine` owns run lifecycle, event streaming, serialized writes,
   checkpoints, persistence, critic-approved memory writeback, and follow-up
   continuity.
2. A typed LangGraph state machine routes through supervisor, researcher,
   summarizer, critic, and writer nodes. Revision loops have explicit budgets.
3. Tavily, arXiv, PDF, critic-approved memory, and bundled-demo sources normalize
   into provenance-labeled `Source` records with stable `S1` identifiers and
   hashes.
4. Deterministic and OpenAI-backed model adapters share validated planning,
   synthesis, critique, and report contracts with bounded prompt payloads.
5. SQLite provides checkpoint, run-ledger, and vector-memory durability. The
   vector store persists deterministic normalized embeddings and ranks with
   in-process cosine similarity; incompatible metadata fails closed.
6. Streamlit owns one process-wide engine and serializes runs. FastAPI and Typer
   reuse the same application facade rather than duplicating workflow behavior.

## Security decision

The initial ChromaDB dependency was removed after the locked dependency audit
reported `PYSEC-2026-311` with no patched release. The replacement is a private
SQLite-backed local vector store that exposes no vector server, uses parameterized
statements, validates schema and embedding metadata, stores files with restrictive
permissions, and preserves durable semantic recall. The final audit reports no
known dependency vulnerabilities.

Untrusted provider, upload, memory, and model text is validated at the source
boundary and escaped or sanitized before Markdown-capable UI rendering. Normal
tests block network sockets; only explicit live and E2E targets enable them.

## Validation record

| Layer | Result | Evidence |
|---|---|---|
| Lock/install | Pass | 140 packages resolved; frozen lock check completed |
| Unit/integration/API/UI/eval | Pass | 127 passed, 3 opt-in tests deselected |
| Branch coverage | Pass | 87.67%, above the 80% floor |
| Research evaluation | Pass | 2 deterministic golden cases |
| Browser E2E | Pass | 2/2 local and 2/2 container journeys; desktop plus 390px mobile |
| In-app browser | Pass | Four stages, report, provenance-labeled sources, critic, follow-up, exports |
| Browser health | Pass | Zero warning/error logs, no error overlay, no mobile overflow |
| Static quality | Pass | Ruff format/lint and strict mypy |
| Security | Pass | Bandit clean; pip-audit found no known vulnerabilities |
| Package | Pass | Wheel and source distribution built; wheel installed in a clean virtualenv |
| Compose configuration | Pass | Default and configurable-port configuration parsed |
| Container runtime | Pass | Healthy non-root runtime, hardened flags, restart persistence readback |
| CLI | Pass | Keyless demo produced a four-source cited report |

Exact local commands:

```bash
uv sync --frozen --all-groups
make check
make eval
uv build
uv run research-desk demo
RESEARCH_DESK_BASE_URL=http://127.0.0.1:8504 uv run pytest --force-enable-socket -m e2e tests/e2e -q
docker compose config --quiet
docker compose build
RESEARCH_DESK_UI_PORT=8505 docker compose up -d --force-recreate workbench
RESEARCH_DESK_BASE_URL=http://127.0.0.1:8505 uv run pytest --force-enable-socket -m e2e tests/e2e -q
docker compose restart workbench
docker compose down
```

## Required proof labels

- Validation Environment: macOS local workspace, CPython 3.11, Docker Desktop,
  Codex in-app Chromium, and Playwright Chromium.
- Validation Scope: Complete local package, deterministic graph, API, CLI,
  workbench, persistence, exports, security gates, responsive browser flow,
  hardened container, and restart readback.
- Data Integrity Classification: `demo_fixture` for bundled evidence and
  `accepted_memory` for automatic model-critic-approved memory. Both labels are
  visible in results; neither means live retrieval or human approval.
- Mock/Fixture Usage: Deterministic demo and golden evaluations use bundled dated
  fixtures. Provider unit tests use mocked HTTP. Browser journeys use the bundled
  demo, not live web results.
- Production Validation Status: Not performed. No deployment or production claim
  is made.
- Localhost Validation Integrity: Real Streamlit process, real LangGraph engine,
  real SQLite persistence, real browser interactions, and real exports.
- Container Validation Integrity: Image manifest
  `sha256:2f65529b9f4281c9fa5091f1a9852a4cb6a4f17d63cd470a2dd7cf497ef46e2f`;
  UID/GID `10001`; `/app/data` mode `0700`; read-only root; capabilities
  dropped; `no-new-privileges`; PID limit 256; health endpoint returned `ok`.
- Persistence Readback: Two completed records existed before restart and the same
  count plus latest run ID `0f3cd260-5a31-4425-a06c-1bea166afd7a` read back
  after restart.
- Warning Suppression Status: No application warnings are suppressed. Global
  pytest filters cover generated PyMuPDF SWIG deprecations; the socket contract
  locally filters only the warning it intentionally provokes while asserting the
  blocking exception.

## Proof boundaries

- Source-complete: Yes.
- Deterministic local runtime: Yes.
- Localhost browser: Yes, desktop and mobile.
- Local container: Yes, including hardened runtime and restart readback.
- Live Tavily/arXiv/OpenAI provider calls: Not exercised; tests are opt-in and no
  provider credentials were used.
- Hosted development: Not exercised.
- Production: Not exercised.
- Dark mode: Not claimed; this release intentionally ships one fixed accessible
  light theme.
- Multi-user hosting: Not claimed. Authentication, tenant isolation, retention,
  rate limiting, and networked persistence remain hosting prerequisites.
- Crash recovery/API idempotency: Not claimed. Checkpoints support durable
  same-thread continuity, not crash resume or cross-request deduplication.

## GitHub publication

- Repository: https://github.com/ownasquare/autonomous-research-system
- Owner and visibility: `ownasquare`, public.
- Default and tracked branch: `main` tracking `origin/main` over HTTPS.
- Initial published commit: `9d4f26737bec7d00264fa746684b09d86cdd9582`.
- Remote readback: `gh repo view` reported the expected owner, description,
  public visibility, and `main` default branch; `git ls-remote` independently
  resolved remote `main` to the same initial published commit.
- Publication scope: source and documentation only. GitHub publication is not a
  hosted application deployment and does not change the live-provider, hosted,
  multi-user, or production proof boundaries above.

## Failure-and-repair record

1. `pip-audit` identified an unpatched ChromaDB vulnerability. ChromaDB and its
   transitive dependency set were removed; vector recall moved to SQLite.
2. Browser proof found ambiguous headings and a mobile sidebar overlay. Selectors
   and responsive sidebar behavior were corrected and regression-tested.
3. Audit passes found stale follow-up warnings, evidence-budget unfairness,
   overbroad demo-topic matching, prompt/URL/Markdown injection surfaces,
   arXiv pacing races, and upload aggregate gaps. Each was fixed with focused
   tests before the final broad gate.
4. The first dependency-heavy Docker build was cancelled after its source became
   stale. Dockerfile cache layering was improved, then the final image built
   cleanly from current source.
5. The first container E2E URL resolved to an unrelated local Streamlit app already
   using port 8501. No external process was stopped; Compose gained configurable
   host ports, and the isolated 8505 container journey passed 2/2.

## Operational handoff

Start locally with:

```bash
uv sync --frozen --all-groups
uv run research-desk ui
```

Use `uv run research-desk demo` for a terminal demonstration, or
`uv run research-desk api` for integrations. Copy `.env.example` to `.env`
only when intentionally enabling live providers. Never present demo evidence as
live or production research.
