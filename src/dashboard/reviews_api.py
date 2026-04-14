"""API endpoints for the review & improvement widget."""
from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from src.dashboard.auth import login_required
from src.dashboard.review_tool_map import TOOL_ENDPOINT_MAP, REQUEST_TYPES
from src.data.repository import Repository
from src.utils.logging import get_logger

logger = get_logger(__name__)

VALID_TYPES = {t[0] for t in REQUEST_TYPES}
VALID_STATUSES = {"pending", "in_progress", "resolved", "needs_info", "closed"}
VALID_PRIORITIES = {"low", "medium", "high"}


def _exec(query):
    return query.execute()


def create_reviews_router(repo: Repository) -> APIRouter:
    router = APIRouter()

    @router.post("/reviews")
    @login_required
    async def create_review(request: Request):
        body = await request.json()
        tool = body.get("tool", "").strip()
        rtype = body.get("type", "").strip()
        title = body.get("title", "").strip()

        if not tool or not rtype or not title:
            return JSONResponse({"error": "tool, type, and title are required"}, 400)
        if tool not in TOOL_ENDPOINT_MAP:
            return JSONResponse({"error": f"Invalid tool: {tool}"}, 400)
        if rtype not in VALID_TYPES:
            return JSONResponse({"error": f"Invalid type: {rtype}"}, 400)

        priority = body.get("priority", "medium")
        if priority not in VALID_PRIORITIES:
            priority = "medium"

        row = {
            "user_id": request.session.get("username", "unknown"),
            "tool": tool,
            "endpoint": (body.get("endpoint") or "").strip() or None,
            "type": rtype,
            "title": title,
            "description": (body.get("description") or "").strip() or None,
            "screenshot_url": body.get("screenshot_url") or None,
            "priority": priority,
        }

        try:
            result = await asyncio.to_thread(
                _exec, repo.db.table("review_requests").insert(row)
            )
            return JSONResponse({"request": result.data[0] if result.data else {}})
        except Exception as e:
            logger.error("create_review_failed", error=str(e))
            return JSONResponse({"error": str(e)}, 500)

    @router.get("/reviews")
    @login_required
    async def list_reviews(request: Request):
        status_filter = request.query_params.get("status")
        try:
            q = repo.db.table("review_requests").select("*").order("created_at", desc=False)
            if status_filter and status_filter in VALID_STATUSES:
                q = q.eq("status", status_filter)
            result = await asyncio.to_thread(_exec, q)
            return JSONResponse({"requests": result.data or []})
        except Exception as e:
            logger.error("list_reviews_failed", error=str(e))
            return JSONResponse({"requests": []})

    # Static paths BEFORE parameterized {review_id} routes
    @router.get("/reviews/tool-map")
    @login_required
    async def get_tool_map(request: Request):
        return JSONResponse({
            "tools": TOOL_ENDPOINT_MAP,
            "types": REQUEST_TYPES,
        })

    @router.post("/reviews/upload")
    @login_required
    async def upload_screenshot(request: Request):
        form = await request.form()
        file = form.get("image")
        if not file:
            return JSONResponse({"error": "No image provided"}, 400)

        content_type = getattr(file, "content_type", "") or ""
        if not content_type.startswith("image/"):
            return JSONResponse({"error": "File must be an image"}, 400)

        data = await file.read()
        if len(data) > 5 * 1024 * 1024:
            return JSONResponse({"error": "File too large (max 5MB)"}, 400)

        user_id = request.session.get("username", "unknown")
        ext = content_type.split("/")[-1].split(";")[0]
        if ext not in ("png", "jpeg", "jpg", "gif", "webp"):
            ext = "png"
        path = f"reviews/{user_id}/{uuid.uuid4()}.{ext}"

        try:
            storage = repo.db.client.storage
            bucket = storage.from_("media")
            await asyncio.to_thread(
                bucket.upload, path, data, {"content-type": content_type}
            )
            public_url = bucket.get_public_url(path)
            return JSONResponse({"url": public_url})
        except Exception as e:
            logger.error("upload_screenshot_failed", error=str(e))
            return JSONResponse({"error": str(e)}, 500)

    @router.get("/reviews/{review_id}")
    @login_required
    async def get_review(request: Request, review_id: str):
        try:
            req_result = await asyncio.to_thread(
                _exec,
                repo.db.table("review_requests").select("*").eq("id", review_id).limit(1),
            )
            if not req_result.data:
                return JSONResponse({"error": "Not found"}, 404)

            comments_result = await asyncio.to_thread(
                _exec,
                repo.db.table("review_comments")
                .select("*")
                .eq("request_id", review_id)
                .order("created_at", desc=False),
            )

            row = req_result.data[0]
            row["comments"] = comments_result.data or []
            return JSONResponse({"request": row})
        except Exception as e:
            logger.error("get_review_failed", error=str(e))
            return JSONResponse({"error": str(e)}, 500)

    @router.patch("/reviews/{review_id}")
    @login_required
    async def update_review(request: Request, review_id: str):
        body = await request.json()
        updates: dict = {}

        if "status" in body and body["status"] in VALID_STATUSES:
            new_status = body["status"]
            updates["status"] = new_status
            # resolved_at rule
            if new_status in ("resolved", "closed"):
                updates["resolved_at"] = "now()"
            elif new_status == "pending":
                updates["resolved_at"] = None

        if "priority" in body and body["priority"] in VALID_PRIORITIES:
            updates["priority"] = body["priority"]

        if not updates:
            return JSONResponse({"error": "No valid fields to update"}, 400)

        try:
            # Handle resolved_at=now() — Supabase client doesn't support SQL functions,
            # so we use a two-step approach for resolved_at
            resolved_at_val = updates.pop("resolved_at", "SKIP")
            if resolved_at_val != "SKIP":
                if resolved_at_val is None:
                    updates["resolved_at"] = None
                else:
                    from datetime import datetime, timezone
                    updates["resolved_at"] = datetime.now(timezone.utc).isoformat()

            result = await asyncio.to_thread(
                _exec,
                repo.db.table("review_requests").update(updates).eq("id", review_id),
            )
            return JSONResponse({"request": result.data[0] if result.data else {}})
        except Exception as e:
            logger.error("update_review_failed", error=str(e))
            return JSONResponse({"error": str(e)}, 500)

    @router.post("/reviews/{review_id}/comments")
    @login_required
    async def add_comment(request: Request, review_id: str):
        body = await request.json()
        content = body.get("content", "").strip()
        if not content:
            return JSONResponse({"error": "content is required"}, 400)

        row = {
            "request_id": review_id,
            "author": "user",
            "content": content,
        }

        try:
            result = await asyncio.to_thread(
                _exec, repo.db.table("review_comments").insert(row)
            )
            return JSONResponse({"comment": result.data[0] if result.data else {}})
        except Exception as e:
            logger.error("add_comment_failed", error=str(e))
            return JSONResponse({"error": str(e)}, 500)

    return router
