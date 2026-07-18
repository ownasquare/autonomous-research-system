# Contributing

1. Install `uv` and Python 3.11, 3.12, or 3.13.
2. Run `uv sync --frozen --all-groups`.
3. Create focused tests before changing behavior.
4. Run `make check` and `make eval` before opening a change.
5. Keep live-provider tests opt-in and never commit credentials or private PDFs.

Changes to source normalization, checkpointing, citations, or report integrity
must include a regression test proving the relevant safety boundary.
