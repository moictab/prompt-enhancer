from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from .. import request_log

router = APIRouter(prefix="/api/requests")


@router.get("")
def list_requests(
    limit: int = Query(50, ge=1, le=200),
    before: str | None = None,
    mode: str | None = None,
    errors: bool = False,
):
    if before is not None and not request_log.ID_PATTERN.match(before):
        raise HTTPException(status_code=400, detail="Invalid cursor")
    page = request_log.list_summaries(limit=limit, before=before, mode=mode or None, errors_only=errors)
    return {**page, "storage_bytes": request_log.storage_bytes()}


@router.get("/images/{name}")
def get_request_image(name: str):
    found = request_log.image_file(name)
    if found is None:
        raise HTTPException(status_code=404, detail="Unknown image")
    path, content_type = found
    return FileResponse(path, media_type=content_type)


@router.get("/{request_id}")
def get_request(request_id: str):
    entry = request_log.get(request_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Unknown request")
    return entry
