from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ENVIRONMENT: str = "development"
    API_VERSION: str = "0.1.0"
    LOG_DB_PATH: str = "api_logs.db"
    # Postgres connection string; empty = fall back to SQLite at LOG_DB_PATH
    DATABASE_URL: str = ""
    # If set, /admin/usage requires a matching X-API-Key header
    ADMIN_API_KEY: str = ""

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
