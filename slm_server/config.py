from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_PREFIX = "SLM_"

# Get the absolute path to the project's root directory
# This assumes config.py is in slm_server/
CONFIG_PY_PATH = Path(__file__).resolve()
PROJECT_ROOT = CONFIG_PY_PATH.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH_DEFAULT = str(MODELS_DIR / "Qwen3-0.6B-Q8_0.gguf")


class LoggingSettings(BaseModel):
    verbose: bool = Field(
        True, description="If logging to stdout by uvicorn and cpp llama"
    )


class MetricsSettings(BaseModel):
    enabled: bool = Field(
        True,
        description="If enable metrics to port",
    )


class TraceSettings(BaseModel):
    enabled: bool = Field(True, description="Enable OpenTelemetry tracing")
    service_name: str = Field(
        "slm_server", description="Service Name used in trace provider."
    )
    endpoint: str = Field("", description="Grafana Tempo OTLP endpoint URL")
    username: str = Field("", description="Grafana Tempo basic auth username")
    password: str = Field("", description="Grafana Tempo basic auth password")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=False, env_prefix=ENV_PREFIX)

    model_path: str = Field(MODEL_PATH_DEFAULT, description="Model path for llama_cpp.")
    n_ctx: int = Field(
        4096, description="Maximum context window (input + generated tokens)."
    )
    n_threads: int = Field(
        2, description="Number of OpenMP threads llama‑cpp will spawn."
    )
    seed: int = Field(42, description="Seed to inject for llama_cpp.")
    s_timeout: int = Field(
        1, description="Seconds to wait if undergoing another inference."
    )

    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    tracing: TraceSettings = Field(default_factory=TraceSettings)


def get_settings() -> Settings:
    if not hasattr(get_settings, "_instance"):
        get_settings._instance = Settings()
    return get_settings._instance
