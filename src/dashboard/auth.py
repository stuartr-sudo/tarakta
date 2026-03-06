from __future__ import annotations

from functools import wraps
from typing import Callable

import bcrypt
from fastapi import Request
from fastapi.responses import JSONResponse, RedirectResponse


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plain text password against a bcrypt hash."""
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


def _is_api_request(request: Request) -> bool:
    """Check if this is an API request (expects JSON, not HTML)."""
    path = request.url.path
    accept = request.headers.get("accept", "")
    # API routes live under /api/ prefix, or the client explicitly wants JSON
    return path.startswith("/api/") or "application/json" in accept


def login_required(func: Callable) -> Callable:
    """Decorator that redirects to /login if not authenticated (any role)."""

    @wraps(func)
    async def wrapper(request: Request, *args, **kwargs):
        if not request.session.get("authenticated"):
            if _is_api_request(request):
                return JSONResponse(
                    {"error": "Not authenticated", "redirect": "/login"},
                    status_code=401,
                )
            return RedirectResponse(url="/login", status_code=303)
        return await func(request, *args, **kwargs)

    return wrapper


def admin_required(func: Callable) -> Callable:
    """Decorator that restricts access to admin role only."""

    @wraps(func)
    async def wrapper(request: Request, *args, **kwargs):
        if not request.session.get("authenticated"):
            if _is_api_request(request):
                return JSONResponse(
                    {"error": "Not authenticated", "redirect": "/login"},
                    status_code=401,
                )
            return RedirectResponse(url="/login", status_code=303)
        if request.session.get("role") != "admin":
            if _is_api_request(request):
                return JSONResponse(
                    {"error": "Admin access required"},
                    status_code=403,
                )
            return RedirectResponse(url="/", status_code=303)
        return await func(request, *args, **kwargs)

    return wrapper
