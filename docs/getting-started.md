# Getting started

## Requirements

- Python 3.11, 3.12, or 3.13
- `uv`
- Docker Desktop only for container validation

## Keyless demo

```bash
uv sync --frozen --all-groups
uv run research-desk demo
uv run research-desk ui
```

The `demo` command uses only a bundled, dated source set labeled `demo_fixture`.
Its purpose is to prove orchestration, citations, persistence, exports, and UI
behavior without spending provider credits or representing fixtures as current
web research. The workbench can separately opt into accepted local memory.

## Live mode

Copy `.env.example` to `.env`, set `RESEARCH_MODE=live`, and provide the provider
keys needed for the selected source/model path. Tavily requires `TAVILY_API_KEY`;
the OpenAI model adapter requires `OPENAI_API_KEY`. arXiv does not require a key.
Do not commit `.env`.

Live search providers receive the generated research queries. If the OpenAI model
adapter is selected, bounded evidence excerpts are also sent to OpenAI; this can
include extracted text from uploaded PDFs and recalled local memory. Review
provider terms and data-handling requirements before using sensitive material.

Live runs never fall back to demo fixtures. If one provider fails, Research Desk
preserves successful live or uploaded evidence and records the coverage loss.

## Surfaces

```bash
uv run research-desk ui
uv run research-desk api
uv run research-desk research "A bounded research topic"
uv run research-desk history
uv run research-desk graph
```

The workbench defaults to `http://127.0.0.1:8501`; the API defaults to
`http://127.0.0.1:8000`.
