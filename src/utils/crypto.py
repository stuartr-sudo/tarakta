"""Symmetric encryption helpers for API key storage."""
from __future__ import annotations

import base64
import hashlib
import os

from cryptography.fernet import Fernet


def _get_fernet() -> Fernet:
    """Derive a Fernet key from SESSION_SECRET env var."""
    secret = os.environ.get("SESSION_SECRET", "fallback-key-not-for-production")
    key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest())
    return Fernet(key)


def encrypt_key(plain: str) -> str:
    """Encrypt an API key for DB storage."""
    return _get_fernet().encrypt(plain.encode()).decode()


def decrypt_key(encrypted: str) -> str:
    """Decrypt an API key from DB storage."""
    return _get_fernet().decrypt(encrypted.encode()).decode()
