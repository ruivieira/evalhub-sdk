"""Command-line interface for EvalHub SDK."""

import asyncio
import importlib
import sys
from pathlib import Path

# typing imports removed - using PEP 604 union syntax
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .adapter.client import AdapterClient, AdapterDiscovery
from .adapter.models import AdapterConfig
from .adapter.server import run_adapter_server

app = typer.Typer(
    name="evalhub-adapter",
    help="EvalHub SDK command-line interface for framework adapters",
    no_args_is_help=True,
)

console = Console()
err_console = Console(file=sys.stderr)


@app.command()
def run(
    adapter_module: str = typer.Argument(
        ...,
        help="Python module path to your adapter class (e.g., 'my_adapter:MyAdapter')",
    ),
    config_file: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file (YAML or JSON)",
        exists=True,
    ),
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to bind to"),
    workers: int = typer.Option(1, help="Number of worker processes"),
    reload: bool = typer.Option(False, help="Enable auto-reload for development"),
    log_level: str = typer.Option("INFO", help="Log level"),
) -> None:
    """Run a framework adapter server.

    Example:
        evalhub-adapter run my_framework.adapter:MyFrameworkAdapter --port 8080
    """
    try:
        # Parse adapter module and class
        if ":" not in adapter_module:
            err_console.print(
                "[red]Error:[/red] Adapter module must be in format 'module:class'"
            )
            raise typer.Exit(1)

        module_path, class_name = adapter_module.split(":", 1)

        # Import the adapter class
        try:
            module = importlib.import_module(module_path)
            adapter_class = getattr(module, class_name)
        except ImportError as e:
            err_console.print(f"[red]Error:[/red] Failed to import {module_path}: {e}")
            raise typer.Exit(1)
        except AttributeError:
            err_console.print(
                f"[red]Error:[/red] Class {class_name} not found in {module_path}"
            )
            raise typer.Exit(1)

        # Load configuration
        config = AdapterConfig(
            framework_id="custom_framework",  # Default, should be overridden
            adapter_name="Custom Framework Adapter",
            version="1.0.0",
            host=host,
            port=port,
            workers=workers,
            max_concurrent_jobs=5,
            job_timeout_seconds=3600,
            memory_limit_gb=None,
            log_level=log_level.upper(),
            enable_metrics=True,
            health_check_interval=30,
        )

        if config_file:
            config = load_config(config_file, config)

        # Display startup information
        console.print(
            Panel.fit(
                f"[bold blue]EvalHub Framework Adapter[/bold blue]\n\n"
                f"[bold]Framework:[/bold] {config.framework_id}\n"
                f"[bold]Adapter:[/bold] {config.adapter_name}\n"
                f"[bold]Version:[/bold] {config.version}\n"
                f"[bold]Server:[/bold] http://{host}:{port}\n"
                f"[bold]Workers:[/bold] {workers}\n"
                f"[bold]Reload:[/bold] {reload}",
                title="Starting Server",
                border_style="blue",
            )
        )

        console.print(f"📚 API Documentation: http://{host}:{port}/docs")
        console.print(f"🏥 Health Check: http://{host}:{port}/api/v1/health")
        console.print(f"ℹ️  Framework Info: http://{host}:{port}/api/v1/info")
        console.print()

        # Run the server
        run_adapter_server(
            adapter_class,
            config,
            reload=reload,
        )

    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped by user[/yellow]")
    except Exception as e:
        err_console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def info(
    url: str = typer.Argument(..., help="Adapter URL (e.g., http://localhost:8080)")
) -> None:
    """Get information about a framework adapter."""

    async def get_info() -> None:
        try:
            async with AdapterClient(url) as client:
                # Get framework info
                framework_info = await client.get_framework_info()

                # Display framework information
                console.print(
                    Panel.fit(
                        f"[bold blue]{framework_info.name}[/bold blue]\n\n"
                        f"[bold]Framework ID:[/bold] {framework_info.framework_id}\n"
                        f"[bold]Version:[/bold] {framework_info.version}\n"
                        f"[bold]Description:[/bold] {framework_info.description or 'N/A'}\n"
                        f"[bold]Benchmarks:[/bold] {len(framework_info.supported_benchmarks)}\n"
                        f"[bold]Model Types:[/bold] {', '.join(framework_info.supported_model_types) or 'N/A'}",
                        title="Framework Information",
                        border_style="blue",
                    )
                )

                # Display benchmarks
                if framework_info.supported_benchmarks:
                    table = Table(title="Available Benchmarks")
                    table.add_column("ID", style="cyan")
                    table.add_column("Name")
                    table.add_column("Category", style="green")
                    table.add_column("Metrics", style="yellow")

                    for benchmark in framework_info.supported_benchmarks[
                        :10
                    ]:  # Show first 10
                        table.add_row(
                            benchmark.benchmark_id,
                            benchmark.name or "N/A",
                            benchmark.category or "N/A",
                            ", ".join(benchmark.metrics)
                            if benchmark.metrics
                            else "N/A",
                        )

                    console.print(table)

                    if len(framework_info.supported_benchmarks) > 10:
                        console.print(
                            f"... and {len(framework_info.supported_benchmarks) - 10} more benchmarks"
                        )

        except Exception as e:
            err_console.print(f"[red]Error:[/red] Failed to get adapter info: {e}")
            raise typer.Exit(1)

    asyncio.run(get_info())


@app.command()
def health(
    url: str = typer.Argument(..., help="Adapter URL (e.g., http://localhost:8080)")
) -> None:
    """Check the health of a framework adapter."""

    async def check_health() -> None:
        try:
            async with AdapterClient(url) as client:
                health_response = await client.health_check()

                # Determine status color
                status_color = {
                    "healthy": "green",
                    "unhealthy": "red",
                    "degraded": "yellow",
                }.get(health_response.status, "white")

                # Display health information
                console.print(
                    Panel.fit(
                        f"[bold {status_color}]{health_response.status.upper()}[/bold {status_color}]\n\n"
                        f"[bold]Framework:[/bold] {health_response.framework_id}\n"
                        f"[bold]Version:[/bold] {health_response.version}\n"
                        f"[bold]Uptime:[/bold] {health_response.uptime_seconds or 0:.1f}s",
                        title="Health Status",
                        border_style=status_color,
                    )
                )

                # Display dependencies
                if health_response.dependencies:
                    table = Table(title="Dependencies")
                    table.add_column("Component", style="cyan")
                    table.add_column("Status")
                    table.add_column("Details")

                    for name, info in health_response.dependencies.items():
                        status = info.get("status", "unknown")
                        status_style = "green" if status == "healthy" else "red"

                        details = []
                        for key, value in info.items():
                            if key != "status":
                                details.append(f"{key}: {value}")

                        table.add_row(
                            name,
                            f"[{status_style}]{status}[/{status_style}]",
                            ", ".join(details) if details else "N/A",
                        )

                    console.print(table)

        except Exception as e:
            err_console.print(f"[red]Error:[/red] Failed to check health: {e}")
            raise typer.Exit(1)

    asyncio.run(check_health())


@app.command()
def discover(
    urls: list[str] = typer.Argument(
        ...,
        help="Adapter URLs to discover (e.g., http://localhost:8080 http://localhost:8081)",
    ),
) -> None:
    """Discover and display information about multiple adapters."""

    async def discover_adapters() -> None:
        discovery = AdapterDiscovery()

        console.print("[blue]Discovering adapters...[/blue]")

        # Discover all adapters
        adapters = []
        for url in urls:
            console.print(f"  Checking {url}...")
            adapter = await discovery.discover_adapter(url)
            if adapter:
                adapters.append(adapter)

        if not adapters:
            console.print("[red]No adapters discovered[/red]")
            return

        # Display results
        table = Table(title="Discovered Adapters")
        table.add_column("URL", style="cyan")
        table.add_column("Framework ID")
        table.add_column("Name")
        table.add_column("Version")
        table.add_column("Status")

        for adapter in adapters:
            status_style = {
                "healthy": "green",
                "unhealthy": "red",
                "degraded": "yellow",
                "unreachable": "red",
            }.get(adapter.status, "white")

            table.add_row(
                adapter.url,
                adapter.framework_id,
                adapter.name,
                adapter.version,
                f"[{status_style}]{adapter.status}[/{status_style}]",
            )

        console.print(table)

    asyncio.run(discover_adapters())


def load_config(config_file: Path, base_config: AdapterConfig) -> AdapterConfig:
    """Load configuration from file."""
    import json

    import yaml

    try:
        with open(config_file) as f:
            if config_file.suffix.lower() in [".yaml", ".yml"]:
                config_data = yaml.safe_load(f)
            else:
                config_data = json.load(f)

        # Update base config with file data
        return base_config.model_copy(update=config_data)

    except Exception as e:
        err_console.print(f"[red]Error:[/red] Failed to load config file: {e}")
        raise typer.Exit(1)


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
