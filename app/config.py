from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Required — no default so startup fails loudly if the key is missing
    OPENAI_API_KEY: str

    # LangSmith tracing — auto-enabled when LANGCHAIN_TRACING_V2="true".
    # LangChain reads these directly from os.environ at import time, so
    # main.py must push them into os.environ before any langchain import.
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "rag-agent"
    LANGCHAIN_TRACING_V2: str = "false"

    # Chunking parameters — exposed here so they're easy to tune in .env
    # without touching source code
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # ChromaDB — docker-compose overrides this to /app/chroma_db via environment:
    CHROMA_PERSIST_DIR: str = "./chroma_db"

    # OpenAI model IDs
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    CHAT_MODEL: str = "gpt-4o-mini"

    # MCP server URL — docker-compose sets this to http://mcp:9001/mcp using
    # Docker's internal DNS; for local dev it points to localhost
    MCP_SERVER_URL: str = "http://localhost:9001/mcp"


# Module-level singleton — import `settings` everywhere, don't instantiate
# Settings() per request (would re-read .env on every call)
settings = Settings()
