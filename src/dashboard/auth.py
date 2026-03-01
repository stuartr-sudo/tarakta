from __future__ import annotations

from functools import wraps
from typing import Callable

import bcrypt
from fastapi import Request
from fastapi.responses import RedirectResponse


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plain text password against a bcrypt hash."""
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


def login_required(func: Callable) -> Callable:
    """Decorator that redirects to /login if not authenticated."""

    @wraps(func)
    async def wrapper(request: Request, *args, **kwargs):
        if not request.session.get("authenticated"):
            return RedirectResponse(url="/login", status_code=303)
        return await func(request, *args, **kwargs)

    return wrapper
