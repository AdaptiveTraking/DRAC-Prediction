from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager

from backend.app.api.endpoints import router as detection_router

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
app.include_router(detection_router, prefix="/drones", tags=["Detection"])
