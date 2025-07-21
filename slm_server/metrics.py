from typing import Callable

import psutil
from fastapi import FastAPI
from prometheus_client import Gauge
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_fastapi_instrumentator.metrics import Info

from slm_server.config import MetricsSettings


def system_cpu_usage() -> Callable[[Info], None]:
    """Custom metric for CPU usage."""
    CPU_USAGE = Gauge("process_cpu_usage_percent", "Current CPU usage in percent")

    def instrumentation(info: Info) -> None:
        CPU_USAGE.set(psutil.cpu_percent())

    return instrumentation


def system_memory_usage() -> Callable[[Info], None]:
    """Custom metric for memory usage."""
    MEMORY_USAGE = Gauge("process_memory_usage_bytes", "Current memory usage in bytes")

    def instrumentation(info: Info) -> None:
        MEMORY_USAGE.set(psutil.Process().memory_info().rss)

    return instrumentation


def setup_metrics(app: FastAPI, settings: MetricsSettings):
    """Setup Prometheus metrics for FastAPI app if enabled."""
    if not settings.enabled:
        return

    instrumentator = Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        should_respect_env_var=False,
        excluded_handlers=[settings.endpoint],
    )

    # Add custom system metrics
    instrumentator.add(system_cpu_usage())
    instrumentator.add(system_memory_usage())

    instrumentator.instrument(app).expose(app, endpoint=settings.endpoint)
