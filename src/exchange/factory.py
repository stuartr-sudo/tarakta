"""Exchange connector factory — registry-based creation of exchange clients.

Connectors register themselves at import time. The factory creates instances
by connector name, abstracting away which class to instantiate.
"""
from __future__ import annotations

from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)

_REGISTRY: dict[str, type] = {}


def register_connector(name: str, cls: type) -> None:
    """Register an exchange connector class under a given name."""
    _REGISTRY[name] = cls
    logger.debug("connector_registered", name=name, cls=cls.__name__)


def create_exchange(connector_name: str, **kwargs: Any):
    """Create an exchange connector by registered name.

    For backward compatibility, also handles the legacy call signature:
        create_exchange("binance", api_key, api_secret, account_type="futures")
    which maps to "binance_spot", "binance_futures", or "binance_margin".
    """
    # Legacy compatibility: "binance" + account_type -> "binance_{account_type}"
    if connector_name == "binance" and "account_type" in kwargs:
        account_type = kwargs.pop("account_type", "spot")
        if account_type == "futures":
            connector_name = "binance_futures"
        elif account_type == "margin":
            connector_name = "binance_margin"
        else:
            connector_name = "binance_spot"

    if connector_name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise ValueError(f"Unknown connector: {connector_name!r}. Available: {available}")

    cls = _REGISTRY[connector_name]
    return cls(**kwargs)


def list_connectors() -> list[str]:
    """Return all registered connector names."""
    return list(_REGISTRY.keys())
