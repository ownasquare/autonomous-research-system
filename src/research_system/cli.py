"""Typer command line interface for research, history, services, and topology."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer
import uvicorn

from research_system.config import Settings
from research_system.engine import build_default_engine
from research_system.exports import export_json
from research_system.graph import graph_mermaid
from research_system.models import (
    ResearchDepth,
    ResearchMode,
    ResearchRequest,
    ResearchResult,
    RunStatus,
)

app = typer.Typer(
    name="research-desk",
    help="Run supervised, source-grounded multi-agent research.",
    no_args_is_help=True,
    add_completion=False,
)


def _print_result(result: ResearchResult, *, as_json: bool = False) -> None:
    if result.status == RunStatus.FAILED:
        typer.echo(result.error or "Research workflow failed.", err=True)
        raise typer.Exit(code=1)
    if as_json:
        typer.echo(export_json(result))
    elif result.report is not None:
        typer.echo(result.report.markdown)


@app.command()
def demo(
    topic: Annotated[
        str,
        typer.Argument(help="Research topic for the bundled, network-free evidence corpus."),
    ] = "How should multi-agent research systems be evaluated?",
) -> None:
    """Run the truthful keyless demo from bundled evidence."""

    settings = Settings(research_mode=ResearchMode.DEMO)
    with build_default_engine(settings) as engine:
        _print_result(
            engine.run(
                ResearchRequest(
                    topic=topic,
                    use_demo=True,
                    use_web=False,
                    use_arxiv=False,
                    use_memory=False,
                )
            )
        )


@app.command()
def research(
    topic: Annotated[str, typer.Argument(help="Question or topic to research.")],
    objective: Annotated[
        str,
        typer.Option(help="What the final report should help the reader accomplish."),
    ] = "Produce a balanced, decision-ready research report.",
    depth: Annotated[
        ResearchDepth,
        typer.Option(help="Source and revision budget."),
    ] = ResearchDepth.STANDARD,
    live: Annotated[
        bool,
        typer.Option(help="Use configured live OpenAI, Tavily, and arXiv providers."),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the full validated result as JSON."),
    ] = False,
) -> None:
    """Run research in demo mode or explicitly opted-in live mode."""

    mode = ResearchMode.LIVE if live else ResearchMode.DEMO
    settings = Settings(research_mode=mode)
    request = ResearchRequest(
        topic=topic,
        objective=objective,
        depth=depth,
        use_demo=not live,
        use_web=live,
        use_arxiv=live,
        use_memory=live,
    )
    with build_default_engine(settings) as engine:
        _print_result(engine.run(request), as_json=json_output)


@app.command()
def ui(
    host: Annotated[str, typer.Option(help="Local interface address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535, help="Local interface port.")] = 8501,
) -> None:
    """Open the Streamlit Research Desk workbench."""

    from streamlit.web import cli as streamlit_cli

    app_path = Path(__file__).with_name("streamlit_app.py")
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        host,
        "--server.port",
        str(port),
    ]
    streamlit_cli.main()


@app.command(name="api")
def api_server(
    host: Annotated[str, typer.Option(help="API bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535, help="API port.")] = 8000,
) -> None:
    """Run the FastAPI integration service."""

    uvicorn.run("research_system.api.app:app", host=host, port=port)


@app.command()
def history(
    limit: Annotated[int, typer.Option(min=1, max=200, help="Maximum runs to show.")] = 20,
) -> None:
    """List durable local research runs without exposing provider secrets."""

    with build_default_engine() as engine:
        runs = engine.list_runs(limit=limit)
    if not runs:
        typer.echo("No saved research runs yet.")
        return
    typer.echo("UPDATED\tSTATUS\tTOPIC\tRUN ID")
    for run in runs:
        typer.echo(
            f"{run.updated_at.isoformat()}\t{run.status.value}\t{run.request.topic}\t{run.run_id}"
        )


@app.command(name="graph")
def show_graph() -> None:
    """Print Mermaid for the executable supervisor/worker topology."""

    with build_default_engine() as engine:
        typer.echo(graph_mermaid(engine.graph))
