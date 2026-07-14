from fastapi import APIRouter

from app.api.routes import bulk_jobs, faces, photos, processes, stats

api_router = APIRouter()
api_router.include_router(faces.router)
api_router.include_router(bulk_jobs.router)
api_router.include_router(processes.router)
api_router.include_router(photos.router)
api_router.include_router(stats.router)
