from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Roboflow cloud settings
    ROBOFLOW_API_KEY: str = "NcZZtYzMxNxEOlCXkJMb"                    # empty default, won't crash
    ROBOFLOW_MODEL_ID: str = "my-first-project-lca2k"
    ROBOFLOW_VERSION: int = 1

    # Detection settings
    CONFIDENCE_THRESHOLD: float = 0.5
    MAX_IMAGE_SIZE_MB: int = 10
    ALLOWED_EXTENSIONS: list = ["jpg", "jpeg", "png", "webp"]

    DEBUG: bool = False                           # False in production

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()