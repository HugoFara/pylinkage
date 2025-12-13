"""Configuration settings for the API."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """API configuration settings."""

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # CORS settings
    cors_origins: list[str] = [
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ]

    # Storage settings
    storage_backend: str = "memory"  # "memory" or "file"
    storage_path: str = "./linkage_data"

    class Config:
        env_prefix = "PYLINKAGE_"
        env_file = ".env"


settings = Settings()
