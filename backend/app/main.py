from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager

from .api.detection import route as detection_route

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Perform any startup tasks here
    print("Starting up DRAC API...")
    yield
    # Perform any shutdown tasks here
    print("Shutting down the DRAC API...")

app = FastAPI(
    title="DRAC prediction API",
    version="1.0.0",
    lifespan=lifespan
)

app.root_path = "/api"

app.include_router(detection_route, prefix="/detection", tags=["Detection"])
