"""Tipos compartidos para la máquina de estados, controlador y módulos de visión."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class State(Enum):
    """Estados principales de la máquina de estados según SPEC.md."""

    IDLE_OR_CAST = "IDLE_OR_CAST"
    WAITING_MINIGAME = "WAITING_MINIGAME"
    SOLVING_DOTS = "SOLVING_DOTS"
    RESOLVING_OUTCOME = "RESOLVING_OUTCOME"
    IN_BATTLE = "IN_BATTLE"
    COOLDOWN = "COOLDOWN"
    STOPPED_ALERT = "STOPPED_ALERT"
    ABORTED = "ABORTED"


class Form(Enum):
    """Formas posibles de un Loomian detectado en combate."""

    NORMAL = "normal"
    CORRUPTO = "corrupto"
    ALPHA = "alpha"
    GAMMA = "gamma"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Point:
    """Coordenada 2D en píxeles."""

    x: int
    y: int


@dataclass(frozen=True, slots=True)
class ROI:
    """Región de interés rectangular en píxeles."""

    x: int
    y: int
    width: int
    height: int

    def to_mss_monitor(self) -> dict[str, int]:
        """Convierte el ROI al formato de diccionario esperado por ``mss``."""
        return {
            "left": self.x,
            "top": self.y,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True, slots=True)
class EncounterResult:
    """Resultado de clasificación de un encuentro o aviso de pesca."""

    name: str | None
    form: Form
    confidence: float
    item_detected: bool = False
    shard: str | None = None
