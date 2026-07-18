"""Portable, safe exports for completed research results."""

from __future__ import annotations

from html import escape
from typing import Literal

from research_system.models import ResearchReport, ResearchResult

ExportFormat = Literal["markdown", "json", "html"]


def _require_report(result: ResearchResult) -> ResearchReport:
    if result.report is None:
        raise ValueError("A completed report is required for export")
    return result.report


def export_markdown(result: ResearchResult) -> str:
    """Return the report's canonical Markdown representation."""

    return _require_report(result).markdown


def export_json(result: ResearchResult) -> str:
    """Return the full validated result with provenance as formatted JSON."""

    return result.model_dump_json(indent=2)


def export_html(result: ResearchResult) -> str:
    """Return a self-contained HTML document with all dynamic text escaped."""

    report = _require_report(result)
    title = escape(report.title)
    markdown = escape(report.markdown)
    provenance = escape(report.provenance_mode.value.replace("_", " ").title())
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{ color-scheme: light dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
    body {{ margin: 0; background: #f3f5f4; color: #17201d; }}
    main {{ max-width: 860px; margin: 2rem auto; padding: 2.5rem; background: #fff;
      border: 1px solid #d9dfdc; border-radius: 16px; box-shadow: 0 12px 36px #17201d14; }}
    h1 {{ line-height: 1.15; }}
    .meta {{ color: #51605b; margin-bottom: 2rem; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; font: inherit; line-height: 1.7; }}
    @media (max-width: 720px) {{ body {{ background: #fff; }} main {{ margin: 0; padding: 1.25rem;
      border: 0; border-radius: 0; box-shadow: none; }} }}
    @media (prefers-color-scheme: dark) {{ body {{ background: #101714; color: #edf4f0; }}
      main {{ background: #17201d; border-color: #30413b; }} .meta {{ color: #aebdb7; }} }}
  </style>
</head>
<body>
  <main>
    <h1>{title}</h1>
    <p class="meta">Run {escape(result.run_id)} · {provenance} evidence</p>
    <pre>{markdown}</pre>
  </main>
</body>
</html>
"""


def export_result(result: ResearchResult, export_format: ExportFormat) -> str:
    """Dispatch an export using the stable public format names."""

    exporters = {
        "markdown": export_markdown,
        "json": export_json,
        "html": export_html,
    }
    try:
        exporter = exporters[export_format]
    except KeyError as exc:
        raise ValueError(f"Unsupported export format: {export_format}") from exc
    return exporter(result)
