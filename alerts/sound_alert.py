"""Alerta sonora para detenciones seguras (STOPPED_ALERT) con fallback sin WAV."""

from __future__ import annotations

import logging
from pathlib import Path

try:
    import winsound
except ImportError:  # pragma: no cover - solo en plataformas no Windows
    winsound = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class SoundAlert:
    """Reproduce una alerta sonora mediante archivo WAV o tono del sistema."""

    def __init__(
        self,
        wav_path: str | Path | None = None,
        beep_frequency: int = 1000,
        beep_duration_ms: int = 500,
        repeat: int = 2,
    ) -> None:
        self.wav_path = Path(wav_path) if wav_path is not None else None
        self.beep_frequency = beep_frequency
        self.beep_duration_ms = beep_duration_ms
        self.repeat = max(1, repeat)

    def trigger(self, reason: str = "") -> None:
        """Dispara la alerta sonora; usa tono del sistema si no existe el WAV."""
        logger.warning("ALERTA SONORA disparada. Motivo: %s", reason or "sin especificar")

        if self.wav_path is not None and self.wav_path.is_file():
            if self._play_wav(self.wav_path):
                return

        self._play_fallback_beep()

    def _play_wav(self, path: Path) -> bool:
        """Intenta reproducir un archivo WAV con winsound."""
        if winsound is None:
            logger.debug("winsound no disponible; usando fallback.")
            return False
        try:
            winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
            return True
        except RuntimeError as exc:
            logger.warning("No se pudo reproducir WAV %s: %s. Usando fallback.", path, exc)
            return False

    def _play_fallback_beep(self) -> None:
        """Emite un tono del sistema (winsound.Beep o campana ASCII)."""
        if winsound is not None:
            try:
                for _ in range(self.repeat):
                    winsound.Beep(self.beep_frequency, self.beep_duration_ms)
                return
            except RuntimeError as exc:
                logger.warning("winsound.Beep falló: %s. Usando campana de consola.", exc)
        print("\a", end="", flush=True)


def play_alert(wav_path: str | Path | None = None, reason: str = "") -> None:
    """Atajo funcional para disparar una alerta sonora inmediata."""
    SoundAlert(wav_path=wav_path).trigger(reason=reason)
