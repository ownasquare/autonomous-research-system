"""Pure presentation helpers shared by the Streamlit workbench and UI tests."""

from __future__ import annotations

import json
import re
from html import escape

from research_system.models import ResearchResult

STAGE_LABELS = {
    "gathering": "Gathering",
    "organizing": "Organizing",
    "reviewing": "Reviewing",
    "writing": "Writing",
    "complete": "Complete",
    "failed": "Stopped",
}

STAGE_PROGRESS = {
    "gathering": 0.2,
    "organizing": 0.5,
    "reviewing": 0.75,
    "writing": 0.9,
    "complete": 1.0,
    "failed": 1.0,
}

FORMAT_SUFFIXES = {"markdown": "md", "json": "json", "html": "html"}


def stage_label(stage: str) -> str:
    """Return restrained, user-facing language for a workflow stage."""

    return STAGE_LABELS.get(stage, "Preparing")


def stage_progress(stage: str) -> float:
    """Return a stable progress value for known workflow stages."""

    return STAGE_PROGRESS.get(stage, 0.0)


def downloadable_filename(run_id: str, format_name: str) -> str:
    """Create a predictable filename without path or shell-significant characters."""

    safe_id = re.sub(r"[^a-zA-Z0-9._-]+", "-", run_id).strip("-._")
    suffix = FORMAT_SUFFIXES[format_name]
    stem = f"research-{safe_id}" if safe_id else "research-report"
    return f"{stem}.{suffix}"


def fallback_markdown_export(result: ResearchResult) -> str:
    """Build a Markdown export if the backend export module is unavailable."""

    if result.report is None:
        return f"# Research run\n\nStatus: {result.status.value}\n"
    return result.report.markdown


def fallback_json_export(result: ResearchResult) -> str:
    """Build a stable, UTF-8 JSON representation of a research run."""

    return json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False)


def fallback_html_export(result: ResearchResult) -> str:
    """Build a self-contained escaped HTML export suitable for offline reading."""

    title = result.report.title if result.report is not None else "Research run"
    summary = result.report.executive_summary if result.report is not None else result.status.value
    report = result.report.markdown if result.report is not None else "No report was produced."
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{ margin: 0 auto; max-width: 820px; padding: 3rem 1.25rem; color: #17211f;
      background: #f6f6f2; font: 16px/1.65 system-ui, sans-serif; }}
    article {{ padding: 2rem; border: 1px solid #dfe4e1; border-radius: 12px;
      background: #fff; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; font: inherit; }}
    .summary {{ color: #3e4a47; }}
  </style>
</head>
<body>
  <article>
    <h1>{title}</h1>
    <p class="summary">{summary}</p>
    <pre>{report}</pre>
  </article>
</body>
</html>
""".format(title=escape(title), summary=escape(summary), report=escape(report))
