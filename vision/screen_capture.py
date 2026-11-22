"""Captura de pantalla y ROIs con ``mss`` y rate-limit configurable."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import numpy as np

from core.types import ROI

logger = logging.getLogger(__name__)


class ScreenCapture:
    """Captura frames BGR desde pantalla/ROI respetando un límite de FPS configurable."""

    def __init__(
        self,
        idle_fps: float = 4.0,
        solving_dots_fps: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        fallback_size: tuple[int, int] | None = None,
        default_roi: ROI | None = None,
    ) -> None:
        if idle_fps <= 0 or solving_dots_fps <= 0:
            raise ValueError("idle_fps y solving_dots_fps deben ser mayores a cero.")
        self.idle_fps = idle_fps
        self.solving_dots_fps = solving_dots_fps
        self._active_fps = idle_fps
        self._clock = clock
        self._sleeper = sleeper
        self._fallback_size = fallback_size
        self.default_roi = default_roi
        self._fallback_warned = False
        self._last_capture_ts: float | None = None
        self._sct: object | None = None

    @property
    def active_fps(self) -> float:
        """FPS objetivo actualmente configurado."""
        return self._active_fps

    def set_solving_mode(self, solving: bool) -> None:
        """Cambia entre frecuencia baja (espera) y alta (SOLVING_DOTS)."""
        self._active_fps = self.solving_dots_fps if solving else self.idle_fps

    def _ensure_mss(self) -> object:
        if self._sct is None:
            import mss

            self._sct = mss.mss()
        return self._sct

    def _enforce_rate_limit(self) -> None:
        min_interval = 1.0 / self._active_fps
        now = self._clock()
        if self._last_capture_ts is not None:
            elapsed = now - self._last_capture_ts
            remaining = min_interval - elapsed
            if remaining > 0:
                self._sleeper(remaining)
        self._last_capture_ts = self._clock()

    def grab(self, roi: ROI | None = None) -> np.ndarray:
        """Captura un ROI (o ``default_roi`` / monitor principal) y devuelve un array BGR."""
        self._enforce_rate_limit()
        sct = self._ensure_mss()

        target_roi = roi if roi is not None else self.default_roi
        if target_roi is not None:
            monitor = target_roi.to_mss_monitor()
        else:
            monitor = sct.monitors[1]  # type: ignore[attr-defined]

        try:
            raw = sct.grab(monitor)  # type: ignore[attr-defined]
            bgra = np.asarray(raw, dtype=np.uint8)
            return bgra[:, :, :3].copy()
        except Exception as exc:
            if self._fallback_size is None:
                raise
            if not self._fallback_warned:
                logger.warning(
                    "Captura de pantalla no disponible (%s). "
                    "Usando frame negro sintético %dx%d en modo dry-run.",
                    exc,
                    self._fallback_size[0],
                    self._fallback_size[1],
                )
                self._fallback_warned = True
            width, height = (
                (target_roi.width, target_roi.height)
                if target_roi is not None
                else self._fallback_size
            )
            return np.zeros((height, width, 3), dtype=np.uint8)

    def close(self) -> None:
        """Libera recursos de la sesión ``mss`` si fue inicializada."""
        if self._sct is not None:
            self._sct.close()  # type: ignore[attr-defined]
            self._sct = None
