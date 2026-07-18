# Evaluation

Research quality is tested as a contract, not inferred from fluent prose.

The focused deterministic golden cases run by `make eval` measure:

- required worker trajectory;
- minimum bundled source count;
- registered citation identifiers and reported citation coverage;
- explicit limitations;
- critic coverage, source-quality, and overall scores;

The broader default suite separately tests bounded revisions, source-kind and
provider selection, provider-failure warnings, export provenance, follow-up
continuity, persistence, and UI rendering.

`make eval` uses fixture evidence and a deterministic model. The normal suite
disables network sockets, which prevents accidental live-provider calls. Paid
providers are outside that green claim. `make test-live` and `make test-e2e`
explicitly enable sockets; `tests/live/` remains opt-in and reports its provider,
time, and data-integrity layer separately.
