"""Discovery service for finding and managing framework adapters."""

import asyncio
import logging
from dataclasses import dataclass

# typing imports removed - using PEP 604 union syntax
from ...models.api import FrameworkInfo, HealthResponse
from .adapter_client import AdapterClient

logger = logging.getLogger(__name__)


@dataclass
class AdapterEndpoint:
    """Information about a discovered adapter endpoint."""

    url: str
    framework_id: str
    name: str
    version: str
    status: str  # "healthy", "unhealthy", "unreachable"
    last_checked: float | None = None
    framework_info: FrameworkInfo | None = None
    health_info: HealthResponse | None = None


class AdapterDiscovery:
    """Service for discovering and managing framework adapter endpoints.

    This helps EvalHub automatically discover available framework adapters
    and route requests to the appropriate adapter.
    """

    def __init__(self) -> None:
        """Initialize the discovery service."""
        self._adapters: dict[str, AdapterEndpoint] = {}
        self._check_interval = 30.0  # Health check interval in seconds
        self._running = False
        self._health_check_task: asyncio.Task | None = None

    def register_adapter(self, url: str, framework_id: str | None = None) -> None:
        """Manually register a framework adapter.

        Args:
            url: The adapter's base URL
            framework_id: Optional framework ID (will be discovered if not provided)
        """
        adapter = AdapterEndpoint(
            url=url,
            framework_id=framework_id or f"unknown_{len(self._adapters)}",
            name="Unknown",
            version="unknown",
            status="unknown",
        )

        self._adapters[url] = adapter
        logger.info(f"Registered adapter: {url}")

    def unregister_adapter(self, url: str) -> bool:
        """Unregister a framework adapter.

        Args:
            url: The adapter's base URL

        Returns:
            bool: True if adapter was unregistered
        """
        if url in self._adapters:
            del self._adapters[url]
            logger.info(f"Unregistered adapter: {url}")
            return True
        return False

    async def discover_adapter(self, url: str) -> AdapterEndpoint | None:
        """Discover information about an adapter at the given URL.

        Args:
            url: The adapter's base URL

        Returns:
            AdapterEndpoint: Adapter information, or None if unreachable
        """
        try:
            async with AdapterClient(url, timeout=10.0) as client:
                # Get framework info
                framework_info = await client.get_framework_info()

                # Get health status
                health_info = await client.health_check()

                adapter = AdapterEndpoint(
                    url=url,
                    framework_id=framework_info.framework_id,
                    name=framework_info.name,
                    version=framework_info.version,
                    status=health_info.status,
                    last_checked=asyncio.get_event_loop().time(),
                    framework_info=framework_info,
                    health_info=health_info,
                )

                logger.info(
                    f"Discovered adapter: {framework_info.name} "
                    f"({framework_info.framework_id}) at {url}"
                )

                return adapter

        except Exception as e:
            logger.warning(f"Failed to discover adapter at {url}: {e}")
            return None

    async def check_adapter_health(self, adapter: AdapterEndpoint) -> AdapterEndpoint:
        """Check the health of a specific adapter.

        Args:
            adapter: The adapter to check

        Returns:
            AdapterEndpoint: Updated adapter information
        """
        try:
            async with AdapterClient(adapter.url, timeout=5.0) as client:
                health_info = await client.health_check()

                adapter.status = health_info.status
                adapter.health_info = health_info
                adapter.last_checked = asyncio.get_event_loop().time()

                logger.debug(
                    f"Health check: {adapter.framework_id} is {adapter.status}"
                )

        except Exception as e:
            adapter.status = "unreachable"
            adapter.last_checked = asyncio.get_event_loop().time()
            logger.warning(f"Health check failed for {adapter.framework_id}: {e}")

        return adapter

    async def refresh_all_adapters(self) -> None:
        """Refresh information for all registered adapters."""
        if not self._adapters:
            logger.debug("No adapters registered for health check")
            return

        logger.debug(f"Checking health of {len(self._adapters)} adapters")

        # Check all adapters concurrently
        tasks = []
        for adapter in self._adapters.values():
            task = asyncio.create_task(self.check_adapter_health(adapter))
            tasks.append(task)

        # Wait for all health checks to complete
        await asyncio.gather(*tasks, return_exceptions=True)

        # Log summary
        healthy_count = len(
            [a for a in self._adapters.values() if a.status == "healthy"]
        )
        logger.info(
            f"Health check complete: {healthy_count}/{len(self._adapters)} adapters healthy"
        )

    async def auto_discover_from_config(self, config: dict[str, str]) -> None:
        """Auto-discover adapters from configuration.

        Args:
            config: Dictionary mapping framework_id to URL
        """
        for framework_id, url in config.items():
            logger.info(f"Discovering adapter for {framework_id} at {url}")

            adapter = await self.discover_adapter(url)
            if adapter:
                self._adapters[url] = adapter
            else:
                # Still register even if discovery fails
                self.register_adapter(url, framework_id)

    def get_adapters(
        self, status: str | None = None, framework_id: str | None = None
    ) -> list[AdapterEndpoint]:
        """Get list of registered adapters.

        Args:
            status: Filter by status ("healthy", "unhealthy", "unreachable")
            framework_id: Filter by framework ID

        Returns:
            List[AdapterEndpoint]: Matching adapters
        """
        adapters = list(self._adapters.values())

        if status:
            adapters = [a for a in adapters if a.status == status]

        if framework_id:
            adapters = [a for a in adapters if a.framework_id == framework_id]

        return adapters

    def get_adapter_for_framework(self, framework_id: str) -> AdapterEndpoint | None:
        """Get a healthy adapter for a specific framework.

        Args:
            framework_id: The framework identifier

        Returns:
            AdapterEndpoint: Healthy adapter, or None if not available
        """
        for adapter in self._adapters.values():
            if adapter.framework_id == framework_id and adapter.status == "healthy":
                return adapter

        return None

    def get_healthy_adapters(self) -> list[AdapterEndpoint]:
        """Get all healthy adapters.

        Returns:
            List[AdapterEndpoint]: All healthy adapters
        """
        return [a for a in self._adapters.values() if a.status == "healthy"]

    async def start_health_monitoring(self, interval: float | None = None) -> None:
        """Start continuous health monitoring of adapters.

        Args:
            interval: Health check interval in seconds
        """
        if self._running:
            logger.warning("Health monitoring is already running")
            return

        if interval:
            self._check_interval = interval

        self._running = True
        self._health_check_task = asyncio.create_task(self._health_monitor_loop())
        logger.info(f"Started health monitoring (interval: {self._check_interval}s)")

    async def stop_health_monitoring(self) -> None:
        """Stop health monitoring."""
        self._running = False

        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None

        logger.info("Stopped health monitoring")

    async def _health_monitor_loop(self) -> None:
        """Background loop for health monitoring."""
        while self._running:
            try:
                await self.refresh_all_adapters()
                await asyncio.sleep(self._check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(5.0)  # Short delay before retrying

    async def shutdown(self) -> None:
        """Shutdown the discovery service."""
        await self.stop_health_monitoring()
        self._adapters.clear()
        logger.info("Discovery service shut down")
