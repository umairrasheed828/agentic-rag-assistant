from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    langsmith_api_key: str
    langsmith_tracing: bool = True
    openai_api_key: str | None = None


settings = Settings()
