from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_PREFIX="SLM_"

# Get the absolute path to the project's root directory
# This assumes config.py is in slm_server/
CONFIG_PY_PATH = Path(__file__).resolve()
PROJECT_ROOT = CONFIG_PY_PATH.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH_DEFAULT = str(MODELS_DIR / "Qwen3-0.6B-Q8_0.gguf")

class Settings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=False, env_prefix=ENV_PREFIX)

    model_path: str = Field(MODEL_PATH_DEFAULT, description="Model path for llama_cpp.")
    n_ctx: int = Field(4096, description="Maximum context window (input + generated tokens).")
    n_threads: int = Field(2, description="Number of OpenMP threads llama‑cpp will spawn.")
    seed: int = Field(42, description="Seed to inject for llama_cpp.")
    s_timeout: int = Field(1, description="Seconds to wait if undergoing another inference.")

settings = Settings()
