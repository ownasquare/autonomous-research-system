# Autonomous Multi-Agent Research System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Research Desk, a locally runnable LangGraph research workspace where a supervisor coordinates Researcher, Summarizer, Critic, and Report Writer agents to deliver source-grounded reports with durable memory.

**Status:** Implementation complete; final proof and handoff results are recorded in `docs/autonomous-research-system/2026-07-18-build-completion.md`.

**Architecture:** A typed LangGraph state machine coordinates deterministic or OpenAI-backed worker agents. Tavily, arXiv, uploaded PDFs, SQLite vector memory, SQLite run storage, and LangGraph checkpoints sit behind small interfaces so the default offline demo and the live provider path use the same workflow. Streamlit presents a restrained workbench with research setup, live stage progress, report review, citations, run history, memory, follow-up turns, and exports; FastAPI and Typer expose the same engine for integrations.

**Tech Stack:** Python 3.11-3.13, LangGraph 1.x, LangChain/OpenAI, Tavily, httpx, PyMuPDF, SQLite vector/checkpoint/run storage, FastAPI, Typer, Streamlit, Pydantic, uv, pytest, Ruff, mypy, Bandit, pip-audit, Playwright, Docker.

---

## File map

- `pyproject.toml`, `uv.lock`: package metadata, bounded runtime dependencies, and locked environment.
- `src/research_system/config.py`: environment-backed settings with safe defaults and secret-free diagnostics.
- `src/research_system/models.py`: validated requests, sources, synthesis, critiques, reports, run records, and events.
- `src/research_system/state.py`: typed LangGraph state and reducers.
- `src/research_system/llm.py`: deterministic demo model and optional `ChatOpenAI` structured-output model.
- `src/research_system/tools/`: Tavily, arXiv, PDF, bundled-demo, and source-deduplication adapters.
- `src/research_system/memory/`: deterministic embeddings, persistent SQLite vector memory, and SQLite checkpoint factory.
- `src/research_system/persistence/runs.py`: SQLite run lifecycle and readback.
- `src/research_system/agents/`: supervisor, researcher, summarizer, critic, and writer nodes.
- `src/research_system/graph.py`: graph construction, conditional routing, and Mermaid topology.
- `src/research_system/engine.py`: streaming workflow facade, persistence, follow-up turns, and export integration.
- `src/research_system/api/`: health, research, run detail, and run-list routes.
- `src/research_system/ui/`: Streamlit workbench, accessible theme, and reusable views.
- `src/research_system/cli.py`: `research`, `serve`, `api`, `history`, and `graph` commands.
- `src/research_system/data/demo_sources.json`: truthful bundled evidence for the keyless demo topic.
- `tests/`: unit, integration, contract, evaluation, UI, live-provider, and browser smoke coverage.
- `docs/`: getting started, architecture, evaluation, security, completion record, and handoff evidence.

### Task 1: Scaffold the package and quality contract

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`, `.dockerignore`, `.python-version`, `.env.example`
- Create: `Makefile`, `LICENSE`, `SECURITY.md`, `CONTRIBUTING.md`, `SUPPORT.md`, `CODE_OF_CONDUCT.md`
- Create: `src/research_system/__init__.py`, `src/research_system/py.typed`
- Test: `tests/contract/test_project_contract.py`

- [ ] **Step 1: Write the failing public-project contract**

```python
def test_required_public_files_exist(project_root: Path) -> None:
    required = {"README.md", "LICENSE", "SECURITY.md", ".env.example", "Makefile"}
    assert required <= {path.name for path in project_root.iterdir()}
```

- [ ] **Step 2: Run the contract and confirm it fails before scaffolding**

Run: `uv run pytest tests/contract/test_project_contract.py -q`
Expected: failure listing the missing public files.

- [ ] **Step 3: Add the package, dependency groups, scripts, and static project files**

Use the `src/` layout, Python `>=3.11,<3.14`, Hatchling, an 80% branch-coverage floor, strict mypy, Ruff security rules, and console scripts named `research-desk` and `research-desk-api`.

- [ ] **Step 4: Lock and validate the environment**

Run: `uv lock && uv sync --all-extras --dev`
Expected: a reproducible `uv.lock` and successful editable install.

### Task 2: Define configuration and domain contracts

**Files:**
- Create: `src/research_system/config.py`
- Create: `src/research_system/models.py`
- Create: `src/research_system/state.py`
- Test: `tests/unit/test_config.py`
- Test: `tests/unit/test_models.py`

- [ ] **Step 1: Write failing validation tests**

```python
def test_research_request_rejects_blank_topic() -> None:
    with pytest.raises(ValidationError):
        ResearchRequest(topic="   ")

def test_settings_diagnostics_do_not_expose_keys(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, openai_api_key="secret")
    assert "secret" not in repr(settings)
```

- [ ] **Step 2: Implement typed settings and immutable domain models**

Model source identifiers as stable `S1`-style labels, restrict research depth and source count, cap PDF bytes/pages, keep provider keys as `SecretStr`, and represent every report with citations, source inventory, critique score, provenance mode, and timestamps.

- [ ] **Step 3: Add the LangGraph state**

Use a `TypedDict` with a reducer-backed conversation list plus request, source, synthesis, critique, report, route, iteration, trace, warning, and error fields.

- [ ] **Step 4: Run the focused tests**

Run: `uv run pytest tests/unit/test_config.py tests/unit/test_models.py -q`
Expected: all configuration and validation tests pass.

### Task 3: Implement source acquisition and PDF ingestion

**Files:**
- Create: `src/research_system/tools/base.py`
- Create: `src/research_system/tools/tavily_search.py`
- Create: `src/research_system/tools/arxiv_search.py`
- Create: `src/research_system/tools/pdf_parser.py`
- Create: `src/research_system/tools/demo_search.py`
- Create: `src/research_system/tools/source_pipeline.py`
- Create: `src/research_system/data/demo_sources.json`
- Test: `tests/unit/tools/test_tavily_search.py`
- Test: `tests/unit/tools/test_arxiv_search.py`
- Test: `tests/unit/tools/test_pdf_parser.py`
- Test: `tests/integration/test_source_pipeline.py`

- [ ] **Step 1: Write response, retry, deduplication, and upload-boundary tests**

```python
def test_pdf_parser_rejects_oversized_upload(parser: PdfParser) -> None:
    with pytest.raises(SourceValidationError, match="size limit"):
        parser.parse(b"%PDF" + b"x" * 2_000_000, filename="large.pdf")

def test_pipeline_deduplicates_canonical_urls(pipeline: SourcePipeline) -> None:
    sources = pipeline.deduplicate([source_a, source_a.model_copy(update={"id": "S9"})])
    assert [source.id for source in sources] == ["S1"]
```

- [ ] **Step 2: Implement network adapters with explicit timeouts and bounded retries**

Tavily activates only when its key is configured. The arXiv adapter uses the official export API with a descriptive user agent, response-status checks, XML parsing, and bounded results. Errors become typed warnings so one source provider cannot erase successful evidence from another.

- [ ] **Step 3: Implement safe PDF parsing**

Open bytes with PyMuPDF, enforce byte/page/text limits, reject encrypted or empty documents, normalize text, and convert accepted uploads into first-class cited sources.

- [ ] **Step 4: Add the truthful keyless demo corpus**

Bundle a small dated corpus about agent orchestration and research quality. Mark every bundled result `demo_fixture`; never label it live web evidence.

- [ ] **Step 5: Run tool and integration tests without network access**

Run: `uv run pytest tests/unit/tools tests/integration/test_source_pipeline.py -q`
Expected: mocked HTTP and local-PDF cases pass with no live provider calls.

### Task 4: Build short- and long-term memory plus run persistence

**Files:**
- Create: `src/research_system/memory/embeddings.py`
- Create: `src/research_system/memory/vector_store.py`
- Create: `src/research_system/memory/checkpoints.py`
- Create: `src/research_system/persistence/runs.py`
- Test: `tests/unit/test_embeddings.py`
- Test: `tests/integration/test_vector_memory.py`
- Test: `tests/integration/test_run_repository.py`

- [ ] **Step 1: Write failing persistence/readback tests**

```python
def test_completed_run_round_trips(repository: RunRepository, completed_run: RunRecord) -> None:
    repository.save(completed_run)
    assert repository.get(completed_run.id) == completed_run

def test_vector_memory_survives_reopen(tmp_path: Path, report: ResearchReport) -> None:
    VectorMemory(tmp_path).remember(report)
    assert VectorMemory(tmp_path).search(report.topic, limit=1)[0].run_id == report.run_id
```

- [x] **Step 2: Implement deterministic local embeddings and SQLite vector persistence**

Hash normalized tokens into a fixed vector, L2-normalize it, persist vectors and report metadata in a private SQLite-backed vector store, and rank candidates by cosine similarity. This keeps the keyless demo local and repeatable without a networked server surface.

- [ ] **Step 3: Implement short-term graph checkpoints**

Compile the application graph with `SqliteSaver`, keep its connection alive for the engine lifetime, key every invocation by `thread_id`, and retain conversation turns across restarts. Use `InMemorySaver` only in isolated graph tests and enable strict checkpoint message-pack handling.

- [ ] **Step 4: Implement SQLite lifecycle persistence**

Use parameterized statements and transactions for `pending`, `running`, `completed`, and `failed` records. Persist request, result, warnings, trace, and error readback without storing provider keys.

- [ ] **Step 5: Run the persistence tests**

Run: `uv run pytest tests/unit/test_embeddings.py tests/integration/test_vector_memory.py tests/integration/test_run_repository.py -q`
Expected: all reopen, update, and isolation tests pass.

### Task 5: Implement model adapters and worker agents

**Files:**
- Create: `src/research_system/prompts.py`
- Create: `src/research_system/llm.py`
- Create: `src/research_system/agents/supervisor.py`
- Create: `src/research_system/agents/researcher.py`
- Create: `src/research_system/agents/summarizer.py`
- Create: `src/research_system/agents/critic.py`
- Create: `src/research_system/agents/writer.py`
- Test: `tests/unit/agents/test_supervisor.py`
- Test: `tests/unit/agents/test_workers.py`
- Test: `tests/unit/test_llm.py`

- [ ] **Step 1: Write deterministic worker-contract tests**

```python
def test_supervisor_routes_in_required_order() -> None:
    assert choose_route(empty_state()) == "researcher"
    assert choose_route(state_with_sources()) == "summarizer"
    assert choose_route(state_with_synthesis()) == "critic"
    assert choose_route(state_with_critique()) == "writer"
```

- [ ] **Step 2: Implement a provider-neutral model protocol**

Expose structured methods for query planning, evidence synthesis, critique, report writing, and follow-up context. The OpenAI adapter uses `ChatOpenAI.with_structured_output`; the deterministic adapter generates source-grounded output from the same validated models.

- [ ] **Step 3: Implement each worker as one focused LangGraph node**

The Researcher plans and acquires evidence, Summarizer builds claim/source mappings, Critic scores coverage and identifies gaps, and Report Writer produces Markdown with only known source IDs. Every node appends a timestamped trace event.

- [ ] **Step 4: Enforce citation integrity**

Reject unknown citation IDs, calculate citation coverage from substantive paragraphs, append a normalized reference list, and preserve critic warnings in the exported report.

- [ ] **Step 5: Run agent tests**

Run: `uv run pytest tests/unit/agents tests/unit/test_llm.py -q`
Expected: required order, revision routing, citation guard, and deterministic outputs pass.

### Task 6: Assemble the LangGraph workflow and application facades

**Files:**
- Create: `src/research_system/graph.py`
- Create: `src/research_system/engine.py`
- Create: `src/research_system/exports.py`
- Create: `src/research_system/api/app.py`
- Create: `src/research_system/api/routes.py`
- Create: `src/research_system/cli.py`
- Create: `src/research_system/__main__.py`
- Create: `langgraph.json`
- Test: `tests/integration/test_graph.py`
- Test: `tests/integration/test_engine.py`
- Test: `tests/api/test_api.py`
- Test: `tests/unit/test_exports.py`

- [ ] **Step 1: Write the failing end-to-end graph test**

```python
def test_demo_graph_produces_cited_report(engine: ResearchEngine) -> None:
    result = engine.run(ResearchRequest(topic="How should multi-agent research be evaluated?"))
    assert result.status == RunStatus.COMPLETED
    assert result.report is not None
    assert "[S1]" in result.report.markdown
    assert [event.agent for event in result.trace][-4:] == [
        "researcher", "summarizer", "critic", "writer"
    ]
```

- [ ] **Step 2: Build the explicit supervisor loop**

Connect `START -> supervisor`, route conditionally to the four worker nodes, return each worker to the supervisor, and route the completed state to `END`. Permit one critic-requested research revision in deep mode.

- [ ] **Step 3: Add the streaming engine and exports**

Yield stable stage events, update SQLite at every lifecycle boundary, save completed report memory, and export Markdown, JSON, and a self-contained HTML document.

- [ ] **Step 4: Expose API and CLI surfaces**

Provide `GET /health`, `POST /research`, `GET /runs`, and `GET /runs/{run_id}` plus equivalent CLI commands. Return typed 4xx errors for invalid input and sanitized 5xx errors for workflow failures.

- [ ] **Step 5: Run graph/API/CLI tests**

Run: `uv run pytest tests/integration/test_graph.py tests/integration/test_engine.py tests/api tests/unit/test_exports.py -q`
Expected: demo run, follow-up thread, failure persistence, API, and exports pass.

### Task 7: Build the Streamlit Research Desk

**Files:**
- Create: `.streamlit/config.toml`
- Create: `src/research_system/streamlit_app.py`
- Create: `src/research_system/ui/theme.py`
- Create: `src/research_system/ui/components.py`
- Create: `src/research_system/ui/app.py`
- Test: `tests/ui/test_app.py`
- Test: `tests/ui/test_components.py`

- [ ] **Step 1: Write Streamlit AppTest coverage for the primary workflow**

```python
def test_workspace_renders_primary_research_controls(app: AppTest) -> None:
    app.run()
    assert app.title[0].value == "Research Desk"
    assert app.text_area(key="research_topic")
    assert app.button(key="start_research")
```

- [ ] **Step 2: Implement a familiar workbench shell**

Use a left sidebar for New research, recent runs, memory, and settings. Keep the main surface focused on topic, objective, depth, source mix, PDF uploads, and a single primary action. Put provider state and graph internals in a Details expander.

- [ ] **Step 3: Implement progress and review states**

Show a four-stage status timeline during execution. On completion, display an executive summary, report, source library, critic notes, agent activity, follow-up input, and Markdown/JSON/HTML downloads.

- [ ] **Step 4: Apply the accessible visual contract**

Use neutral ink/canvas/surface tokens, teal action color, 44px controls, visible focus rings, reduced-motion support, responsive stacking, clear empty/error/loading states, and no user content in unsafe HTML.

- [ ] **Step 5: Run UI tests**

Run: `uv run pytest tests/ui -q`
Expected: setup, validation, demo execution, report, history, error, memory, and export states pass.

### Task 8: Add evaluation, operations, CI, and documentation

**Files:**
- Create: `tests/eval/golden_cases.json`
- Create: `tests/eval/test_research_quality.py`
- Create: `tests/live/test_live_providers.py`
- Create: `tests/e2e/test_streamlit_smoke.py`
- Create: `Dockerfile`, `compose.yaml`
- Create: `.github/workflows/ci.yml`
- Create: `README.md`
- Create: `docs/getting-started.md`
- Create: `docs/architecture.md`
- Create: `docs/evaluation.md`
- Create: `docs/security.md`
- Create: `docs/autonomous-research-system/2026-07-18-build-completion.md`

- [ ] **Step 1: Add deterministic quality gates**

Evaluate source count, citation validity, citation coverage, required sections, critic score, stable agent order, and absence of unsupported source IDs across bundled golden cases.

- [ ] **Step 2: Add operational packaging**

Run as a non-root container, persist `/app/data`, health-check Streamlit, keep secrets outside images, and provide Compose profiles for the workbench and API.

- [ ] **Step 3: Document both demo and live modes**

README quick start begins with `uv sync --frozen` and `uv run research-desk demo`. Clearly distinguish bundled-fixture, localhost, live-provider, hosted, and production proof.

- [ ] **Step 4: Run the full local quality matrix**

Run: `make check && make eval && uv build && uv run pip-audit`
Expected: tests, branch coverage, Ruff, formatting, mypy, Bandit, evaluation, package build, and dependency audit all pass.

- [ ] **Step 5: Validate the container**

Run: `docker compose build && docker compose up -d && docker compose ps`
Expected: the workbench reports healthy on its configured localhost port.

### Task 9: Rendered browser proof and completion handoff

**Files:**
- Update: `docs/autonomous-research-system/2026-07-18-build-completion.md`
- Create: `docs/handoffs/2026-07-18-codex-autonomous-research-system.handoff.mdc`
- Create: `/Users/fortunevieyra/Documents/Github/beladed.com/docs/handoffs/2026-07-18-codex-autonomous-research-system.handoff.mdc`

- [ ] **Step 1: Define the target flow**

The flow under test is: Research Desk loads -> user starts the bundled demo -> four agent stages complete -> cited report, source library, critique, follow-up, and downloads render without runtime errors.

- [ ] **Step 2: Exercise desktop and mobile UI through the in-app Browser**

Verify page identity, meaningful DOM, no framework overlay, console health, topic submission, completion state, source expansion, follow-up control, download controls, and responsive layout.

- [ ] **Step 3: Inspect every captured screenshot**

Treat clipping, blank states, unreadable contrast, error banners, overflow, or broken controls as immediate defects; fix and rerun the same interaction before recording proof.

- [ ] **Step 4: Record truthful proof boundaries**

Document local source/test proof, keyless demo fixture proof, localhost browser proof, container proof, and any unexercised live-provider or hosted-production layers separately.

- [ ] **Step 5: Create the required completion and continuation artifacts**

Include architecture, decisions, validation, failures, exact commands, current Git state, risks, prioritized next items, and confidence/freshness tags in the 12-section handoff package.

## Self-review

- Spec coverage: LangGraph supervisor/workers, Tavily, arXiv, PDF parsing, SQLite vector long-term memory, short-term thread checkpoints, Streamlit, citations, follow-up conversation, exports, API, CLI, tests, docs, CI, and containers are each mapped to a task.
- Placeholder scan: all implementation steps name concrete files, behavior, commands, and expected evidence.
- Type consistency: `ResearchRequest`, `Source`, `Synthesis`, `Critique`, `ResearchReport`, `RunRecord`, `WorkflowEvent`, and `ResearchState` are the canonical contracts used across tools, agents, engine, API, CLI, UI, persistence, and tests.
- Licensing consistency: because PyMuPDF is AGPL-3.0 or commercially licensed, this repository ships under AGPL-3.0 and documents the commercial-license alternative rather than presenting the dependency stack as permissively licensed.
