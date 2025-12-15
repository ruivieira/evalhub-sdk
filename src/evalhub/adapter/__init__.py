"""Adapter SDK components for building and hosting framework adapters."""

# Core adapter building components
from .api.endpoints import create_adapter_api
from .api.router import AdapterAPIRouter

# Client components for communicating with adapters
from .client.adapter_client import AdapterClient, ClientError
from .client.discovery import AdapterDiscovery
from .models.framework import AdapterConfig, AdapterMetadata, FrameworkAdapter

# Server components for hosting adapters
from .server.app import AdapterServer

__all__ = [
    # Adapter building
    "AdapterConfig",
    "AdapterMetadata",
    "FrameworkAdapter",
    # Server hosting
    "AdapterServer",
    "create_adapter_api",
    "AdapterAPIRouter",
    # Client communication
    "AdapterClient",
    "ClientError",
    "AdapterDiscovery",
]
