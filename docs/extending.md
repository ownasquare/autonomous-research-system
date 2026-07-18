# Extending Research Desk

## Add a search provider

Implement the normalized search-provider protocol, return validated `Source`
records, and add tests for status failures, timeouts, retry ceilings, URL
normalization, deduplication, and integrity labels. SDK response objects must not
enter graph state.

## Add a model provider

Implement the structured model gateway for query planning, synthesis, critique,
and writing. Preserve schema validation, one bounded repair attempt, provider
timeouts, and citation post-validation. Do not add a second conversation-memory
mechanism beside LangGraph checkpoints.

## Change embeddings

Create a new versioned SQLite-backed vector store or implement an explicit migration
and re-embedding pass. The stored embedding name and dimensions fail closed on
incompatible reuse because mixing vectors invalidates similarity meaning.

## Change the workflow

Update the explicit supervisor allowlist and add an exact route test. Every
cycle must have a code-enforced budget and every terminal report must pass the
citation validator.
