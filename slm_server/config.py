from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_PREFIX = "SLM_"

# Get the absolute path to the project's root directory
# This assumes config.py is in slm_server/
CONFIG_PY_PATH = Path(__file__).resolve()
PROJECT_ROOT = CONFIG_PY_PATH.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DOTENV_PATH = PROJECT_ROOT / ".env"


MODEL_PATH_DEFAULT = str(MODELS_DIR / "Qwen3-0.6B-Q4_K_M.gguf")
MODEL_OWNER_DEFAULT = "second-state"


class LoggingSettings(BaseModel):
    verbose: bool = Field(True, description="If logging to stdout by cpp llama")
    level: str = Field("INFO", description="Log level default for loggers.")


class MetricsSettings(BaseModel):
    enabled: bool = Field(
        True,
        description="If enable metrics to port",
    )
    endpoint: str = Field("/metrics", description="Endpoint of metrics get handler.")


class TraceSettings(BaseModel):
    enabled: bool = Field(True, description="Enable OpenTelemetry tracing")
    service_name: str = Field(
        "slm_server", description="Service Name used in trace provider."
    )
    endpoint: str = Field("", description="Grafana Tempo OTLP endpoint URL")
    username: str = Field("", description="Grafana Tempo basic auth username")
    password: str = Field("", description="Grafana Tempo basic auth password")
    sample_rate: float = Field(
        0.1, description="Trace sampling rate (0.0-1.0), default 10%"
    )
    excluded_urls: list[str] = Field(
        ["/metrics", "/health"],
        description="List of URLs to exclude from tracing",
    )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_prefix=ENV_PREFIX,
        env_nested_delimiter="__",
        env_file=DOTENV_PATH,
        env_file_encoding="utf-8",
    )

    model_path: str = Field(MODEL_PATH_DEFAULT, description="Model path for llama_cpp.")
    model_owner: str = Field(
        MODEL_OWNER_DEFAULT,
        description="Owner label for /models list. Set SLM_MODEL_OWNER to override.",
    )
    n_ctx: int = Field(
        4096, description="Maximum context window (input + generated tokens)."
    )
    n_threads: int = Field(
        2, description="Number of OpenMP threads llama‑cpp will spawn."
    )
    n_batch: int = Field(
        512, description="Number of tokens to process in a single batch."
    )
    seed: int = Field(42, description="Seed to inject for llama_cpp.")
    s_timeout: int = Field(
        1, description="Seconds to wait if undergoing another inference."
    )

    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    metrics: MetricsSettings = Field(default_factory=MetricsSettings)
    tracing: TraceSettings = Field(default_factory=TraceSettings)


def get_settings() -> Settings:
    if not hasattr(get_settings, "_instance"):
        get_settings._instance = Settings()
    return get_settings._instance
