"""Client components for communicating with framework adapters."""

from .adapter_client import AdapterClient
from .discovery import AdapterDiscovery

__all__ = ["AdapterClient", "AdapterDiscovery"]
