from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.routers import detection
from app.config import settings

app = FastAPI(
    title="Wood Log Counter API",
    description="AI-powered wood log counting using YOLOv8",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detection.router, prefix="/api/v1", tags=["Detection"])


@app.get("/", tags=["Health"])
def root():
    return {
        "message": "🪵 Wood Log Counter API is running!",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)