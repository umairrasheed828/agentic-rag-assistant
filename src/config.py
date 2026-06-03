from pydantic_settings import BaseSettings, SettingsConfigDict

from dotenv import load_dotenv

load_dotenv()  # pushes .env values into os.environ so libraries (LangChain) can see them


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    langsmith_api_key: str
    langsmith_tracing: bool = True
    openai_api_key: str | None = None
    database_url: str


settings = Settings()
