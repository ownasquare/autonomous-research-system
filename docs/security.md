# Security and trust model

## Protected assets

Provider credentials, uploaded document contents, private research topics,
checkpoint state, report history, and accepted long-term memories are protected.

## Principal threats and controls

| Threat | Control |
|---|---|
| Prompt injection in sources | Source text is untrusted evidence. It may influence synthesis, critique, revision routing, and follow-up queries, but cannot directly execute code or select arbitrary tool endpoints; structured outputs and source/citation validation remain enforced. |
| Invented citations | Stable source registry plus blocking citation validation. |
| Oversized or hostile PDFs | Byte, page, encryption, and extracted-text limits before graph entry. |
| Server-side request forgery | Users cannot provide arbitrary fetch URLs; providers own fixed endpoints. |
| Secret disclosure | Environment-only keys, `SecretStr`, and key-name-only readiness diagnostics. |
| Duplicate paid calls | Per-run source, prompt, revision, and retry budgets bound exposure. The API does not provide cross-request idempotency, so clients must prevent duplicate submissions. |
| Silent fixture substitution | Demo and live modes are distinct integrity contracts. |
| Persistence leakage | Local data directory is private, ignored, and containerized in a dedicated volume. |
| Checkpoint object revival | Pickle fallback is disabled and MessagePack deserialization uses an explicit safe-type allowlist. |

## PDF licensing

PyMuPDF is dual-licensed under AGPL-3.0 or a commercial license. This repository
uses AGPL-3.0-or-later. A proprietary redistribution must obtain the appropriate
commercial PyMuPDF license and complete its own legal review.

## External provider data flow

Live mode sends research queries to the enabled search providers. When the
OpenAI model adapter is enabled, bounded evidence excerpts are sent to OpenAI for
planning, synthesis, critique, and report writing. Those excerpts can include
web/arXiv results, extracted text from uploaded PDFs, and recalled local research
memory. Do not submit confidential or regulated material without confirming the
applicable provider terms, retention settings, and organizational approval.

## Production boundary

The local SQLite checkpointer, run ledger, and in-process vector index are
appropriate for this single-user portfolio runtime. They are permission-hardened
local files and do not expose a vector database server. A multi-user hosted
deployment must add authentication, tenant isolation, networked persistence,
encryption and backup, retention controls, rate limiting, and provider-specific
compliance review.
