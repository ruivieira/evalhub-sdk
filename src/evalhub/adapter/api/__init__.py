"""API module for EvalHub SDK - Standard REST endpoints."""

from .endpoints import create_adapter_api
from .router import AdapterAPIRouter

__all__ = ["AdapterAPIRouter", "create_adapter_api"]
