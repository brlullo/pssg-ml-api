from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ENVIRONMENT: str = "development"
    API_VERSION: str = "0.1.0"
    LOG_DB_PATH: str = "api_logs.db"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
