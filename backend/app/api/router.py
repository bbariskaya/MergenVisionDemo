from fastapi import APIRouter

from app.api.routes import faces, photos, processes, stats

api_router = APIRouter()
api_router.include_router(faces.router)
api_router.include_router(processes.router)
api_router.include_router(photos.router)
api_router.include_router(stats.router)
