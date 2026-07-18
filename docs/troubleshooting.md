# Troubleshooting

## Demo has no sources

Confirm `RESEARCH_MODE=demo` and that `src/research_system/data/demo_sources.json`
is present. Demo mode does not require a network connection.

## Live web search is unavailable

Tavily requires a nonblank `TAVILY_API_KEY`. A 401 or 403 is a configuration
error and is not retried. Rate limits and transient 5xx responses receive only
bounded retries, after which the run continues with successful live sources and
a visible warning.

## arXiv is slow

Research Desk respects arXiv's requested pacing and caches identical queries.
Do not parallelize arXiv requests or lower the three-second minimum interval.

## PDF has no text

Scanned PDFs can contain images without extractable text. OCR is intentionally
not automatic because it requires a separate Tesseract installation and is much
slower. Convert the document to searchable text or upload a text-bearing PDF.

## Local data is locked

Run only one writer against a given local data directory. If another workbench
owns it, stop that process cleanly before restarting. Do not delete SQLite lock
or journal files while a process is active.
