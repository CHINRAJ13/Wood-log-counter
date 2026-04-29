from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Roboflow cloud settings
    ROBOFLOW_API_KEY: str = "your_api_key_here"
    ROBOFLOW_MODEL_ID: str = "my-first-project-lca2k"
    ROBOFLOW_VERSION: int = 2

    # Detection settings
    CONFIDENCE_THRESHOLD: float = 0.5
    MAX_IMAGE_SIZE_MB: int = 10
    ALLOWED_EXTENSIONS: list = ["jpg", "jpeg", "png", "webp"]

    DEBUG: bool = True

    class Config:
        env_file = ".env"


settings = Settings()