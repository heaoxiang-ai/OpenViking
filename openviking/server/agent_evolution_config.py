# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Live access to the instance-wide Agent Evolution switch."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from openviking_cli.utils.logger import get_logger

logger = get_logger(__name__)


class AgentEvolutionConfigProvider:
    """Read the live Agent Evolution value from the configured ov.conf.

    Kubernetes projects Secret updates into the mounted config directory
    atomically. Reading the file at commit time lets already-created sessions
    observe the new value without restarting the server.
    """

    def __init__(
        self,
        *,
        default_enabled: bool,
        config_path: Optional[str | Path] = None,
    ) -> None:
        self._default_enabled = bool(default_enabled)
        self._last_valid_enabled = bool(default_enabled)
        self._config_path = Path(config_path).expanduser() if config_path else None

    def set_default_enabled(self, enabled: bool) -> None:
        self._default_enabled = bool(enabled)
        self._last_valid_enabled = bool(enabled)

    def is_enabled(self) -> bool:
        config_path = self._config_path
        if config_path is None:
            configured_path = os.getenv("OPENVIKING_CONFIG_FILE", "").strip()
            if not configured_path:
                return self._last_valid_enabled
            config_path = Path(configured_path).expanduser()

        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            enabled = payload["server"]["agent_evolution"]["enabled"]
            if not isinstance(enabled, bool):
                raise ValueError("server.agent_evolution.enabled must be a boolean")
        except OSError as exc:
            logger.warning(
                "Failed to access Agent Evolution config file %s, using last valid value: %s",
                config_path,
                exc,
            )
            return self._last_valid_enabled
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            logger.warning(
                "Failed to read Agent Evolution config from %s, using last valid value: %s",
                config_path,
                exc,
            )
            return self._last_valid_enabled

        self._last_valid_enabled = enabled
        return enabled
