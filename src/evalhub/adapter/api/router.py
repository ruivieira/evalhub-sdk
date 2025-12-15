"""API router utilities for framework adapters."""

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..models.framework import FrameworkAdapter
from .endpoints import create_adapter_api

logger = logging.getLogger(__name__)


class AdapterAPIRouter:
    """Router for creating standardized API servers for framework adapters."""

    def __init__(self, adapter: FrameworkAdapter):
        """Initialize the router with a framework adapter.

        Args:
            adapter: The framework adapter to expose via API
        """
        self.adapter = adapter
        self.app = FastAPI(
            title="EvalHub Framework Adapter",
            description=f"{adapter.config.adapter_name} - API for {adapter.config.framework_id} framework adapter",
            version=adapter.config.version,
            docs_url="/docs",
            redoc_url="/redoc",
        )
        self._setup_middleware()
        self._setup_routes()
        self._setup_exception_handlers()
        self._setup_events()

    def _setup_middleware(self) -> None:
        """Set up middleware for the FastAPI app."""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Note: Request logging removed to avoid middleware compatibility issues

    def _setup_routes(self) -> None:
        """Set up API routes."""
        # Include the standard adapter endpoints
        api_router = create_adapter_api(self.adapter)
        self.app.include_router(
            api_router, prefix="/api/v1", tags=["Framework Adapter API"]
        )

        # Root endpoint
        @self.app.get("/", tags=["Root"])
        async def root() -> dict[str, str]:
            """Root endpoint with basic information."""
            framework_info = await self.adapter.get_framework_info()
            return {
                "message": f"Welcome to {self.adapter.config.adapter_name} API",
                "framework_id": framework_info.framework_id,
                "version": framework_info.version,
                "api_docs": "/docs",
                "health_check": "/api/v1/health",
            }

    def _setup_exception_handlers(self) -> None:
        """Set up global exception handlers."""

        @self.app.exception_handler(404)
        async def not_found_handler(request: Request, exc: Any) -> JSONResponse:
            return JSONResponse(
                status_code=404,
                content={
                    "error_type": "NotFound",
                    "error_message": "Resource not found",
                    "path": str(request.url.path),
                },
            )

        @self.app.exception_handler(500)
        async def internal_error_handler(request: Request, exc: Any) -> JSONResponse:
            logger.exception("Internal server error")
            return JSONResponse(
                status_code=500,
                content={
                    "error_type": "InternalError",
                    "error_message": "An internal error occurred",
                },
            )

    def _setup_events(self) -> None:
        """Set up startup and shutdown event handlers."""

        @self.app.on_event("startup")
        async def startup_event() -> None:
            await self.startup()

        @self.app.on_event("shutdown")
        async def shutdown_event() -> None:
            await self.shutdown()

    async def startup(self) -> None:
        """Startup handler for the API server."""
        try:
            await self.adapter.initialize()
            logger.info(
                f"Framework adapter {self.adapter.config.framework_id} initialized"
            )
        except Exception:
            logger.exception("Failed to initialize framework adapter")
            raise

    async def shutdown(self) -> None:
        """Shutdown handler for the API server."""
        try:
            await self.adapter.shutdown()
            logger.info(
                f"Framework adapter {self.adapter.config.framework_id} shut down"
            )
        except Exception:
            logger.exception("Error during adapter shutdown")

    def get_app(self) -> FastAPI:
        """Get the FastAPI application.

        Returns:
            FastAPI: The configured FastAPI application
        """
        return self.app
