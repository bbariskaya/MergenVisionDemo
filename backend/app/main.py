from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.api import router as api_router
from app.api.routes import health
from app.core.errors import ConflictError, MergenVisionError, NotFoundError, ValidationError
from app.lifespan import lifespan


def create_app() -> FastAPI:
    app = FastAPI(title="MergenVision Demo", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(MergenVisionError)
    async def domain_error_handler(request, exc: MergenVisionError):
        if isinstance(exc, NotFoundError):
            status_code = 404
        elif isinstance(exc, ValidationError):
            status_code = 422
        elif isinstance(exc, ConflictError):
            status_code = 409
        else:
            status_code = 500
        return JSONResponse(status_code=status_code, content={"detail": str(exc)})

    app.include_router(health.router)
    app.include_router(api_router.api_router, prefix="/api/v1")
    return app


app = create_app()
