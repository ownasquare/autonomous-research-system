# Research Desk

Research Desk is a source-grounded research workspace where a LangGraph
supervisor coordinates four focused workers: Researcher, Summarizer, Critic,
and Report Writer. A topic becomes a decision-ready report with inline source
IDs, a reference library, quality review, run history, LangGraph thread memory,
and durable SQLite vector recall.

The default demo is deterministic and needs no credentials. Live mode can add
Tavily web search, the arXiv API, uploaded PDFs, and an OpenAI-compatible model
without changing the workflow.

## Quick start

```bash
uv sync --frozen --all-groups
uv run research-desk demo
uv run research-desk ui
```

Open `http://127.0.0.1:8501` for the workbench. See
`docs/getting-started.md` for live-provider setup and `docs/architecture.md`
for the state machine, memory, persistence, and citation contracts.

## Proof boundary

The bundled demo uses clearly labeled fixture evidence. It proves local graph,
memory, citation, persistence, API, CLI, export, and UI behavior; it is not
presented as live web or production-hosted research. Live-provider checks are
opt-in and reported separately.

## License

AGPL-3.0-or-later. PyMuPDF is AGPL-3.0 or commercially licensed; proprietary
redistribution requires an appropriate commercial license.
