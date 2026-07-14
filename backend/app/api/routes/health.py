from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.services.readiness import ComponentStatus, ReadinessService

router = APIRouter()


@router.get("/health/live")
async def live():
    return {"status": "alive"}


@router.get("/health/ready")
async def ready(request: Request):
    service: ReadinessService = request.app.state.readiness_service
    statuses, is_ready = await service.check()

    components = []
    for s in statuses:
        comp: dict[str, object] = {"name": s.name, "status": s.status}
        if s.details:
            comp["details"] = s.details
        components.append(comp)

    body = {
        "status": "ready" if is_ready else "unavailable",
        "components": components,
    }

    status_code = 200 if is_ready else 503
    return JSONResponse(status_code=status_code, content=body)
