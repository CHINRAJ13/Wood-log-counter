from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Model
    MODEL_PATH: str = "models/wood_logs.pt"
    CONFIDENCE_THRESHOLD: float = 0.5
    IOU_THRESHOLD: float = 0.45
    IMAGE_SIZE: int = 640

    # Upload
    MAX_IMAGE_SIZE_MB: int = 10
    ALLOWED_EXTENSIONS: list = ["jpg", "jpeg", "png", "webp"]

    # App
    DEBUG: bool = True

    class Config:
        env_file = ".env"


settings = Settings()