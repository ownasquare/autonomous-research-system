# Security policy

## Supported version

Security fixes are applied to the latest release on the default branch.

## Reporting

Report a vulnerability privately to the repository owner. Do not include API
keys, uploaded documents, or provider responses in a public issue.

## Security boundaries

- Provider keys are read from the environment and represented as secret values.
- The keyless demo never makes a paid-provider call.
- Uploaded PDFs are size-, page-, and text-limited before they enter the graph.
- Retrieved and uploaded text is untrusted evidence: it may influence model analysis and bounded revision routing, but it cannot directly execute code or choose arbitrary tool endpoints.
- Reports may cite only source identifiers registered by the research pipeline.
- Normal tests block network sockets. Only the explicit `make test-live` target enables sockets for tests marked `live`.

See `docs/security.md` for the threat model and operational controls.
