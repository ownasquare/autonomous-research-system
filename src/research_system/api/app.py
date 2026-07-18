"""FastAPI application factory with owned-engine lifecycle management."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from research_system.api.routes import router
from research_system.config import Settings
from research_system.engine import ResearchEngine, build_default_engine


def create_app(
    engine: ResearchEngine | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    """Create an API without opening provider or persistence resources at import time."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        owns_engine = engine is None
        application.state.engine = engine or build_default_engine(settings)
        try:
            yield
        finally:
            if owns_engine:
                application.state.engine.close()

    application = FastAPI(
        title="Research Desk API",
        version="0.1.0",
        description="Supervised, source-grounded multi-agent research.",
        lifespan=lifespan,
    )
    application.include_router(router)
    return application


app = create_app()


def main() -> None:
    """Run the API console entry point."""

    uvicorn.run("research_system.api.app:app", host="127.0.0.1", port=8000)
