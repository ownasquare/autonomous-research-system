# Operations

## Local process

`research-desk ui` owns one process-wide engine shared by local Streamlit
sessions. Research runs are serialized through that engine so its SQLite-backed
ledger, checkpoints, and memory retain a single-writer boundary. An interpreter
shutdown hook closes the process-owned engine. The data files live beneath
`RESEARCH_DATA_DIR`.

## Container

```bash
docker compose build
docker compose up -d
docker compose ps
docker compose down
```

The default workbench binds only to `127.0.0.1:8501`, runs as UID 10001, drops
Linux capabilities, uses a read-only root filesystem, and persists `/app/data`
in a named volume. Run the API as an alternative single-writer surface with
`docker compose --profile api up -d api`; stop the workbench first. Do not run
the workbench and API against the same SQLite volume concurrently.

If a local service already uses a default port, set `RESEARCH_DESK_UI_PORT` or
`RESEARCH_DESK_API_PORT` before starting Compose; the container ports remain 8501
and 8000 respectively.

## Backup and recovery

Stop every writer before copying data. Back up and restore the complete
`RESEARCH_DATA_DIR` or named volume atomically so the run ledger, checkpoints,
vector memory, and every SQLite WAL/SHM sidecar remain consistent.

## Health

- Workbench: `/_stcore/health` on port 8501
- API: `/health` on port 8000
- Generic service state: Settings page
- Detailed provider readiness: API `/health` payload; key values are never returned
