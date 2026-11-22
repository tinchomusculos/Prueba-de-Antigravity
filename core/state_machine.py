"""Máquina de estados principal para el ciclo de pesca en Loomian Legacy."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from typing import Protocol

import numpy as np

from core.types import EncounterResult, Form, Point, ROI, State

logger = logging.getLogger(__name__)

TERMINAL_STATES = frozenset({State.STOPPED_ALERT, State.ABORTED})


class ScreenCaptureProtocol(Protocol):
    """Interfaz mínima requerida para captura de pantalla."""

    def set_solving_mode(self, solving: bool) -> None:
        """Alterna entre rate-limit de espera y de SOLVING_DOTS."""

    def grab(self, roi: ROI | None = None) -> np.ndarray:
        """Captura un frame BGR."""


class MinigameTrackerProtocol(Protocol):
    """Interfaz mínima requerida para detección y resolución del minijuego."""

    def is_minigame_active(self, frame: np.ndarray) -> bool:
        """Indica si el minijuego está visible."""

    def get_ordered_nodes(self, frame: np.ndarray) -> list[Point]:
        """Devuelve la lista ordenada de coordenadas de los nodos."""


class EncounterClassifierProtocol(Protocol):
    """Interfaz mínima para detectar resultados del minijuego y combates."""

    def detect_item_notice(self, frame: np.ndarray) -> bool:
        """Detecta el aviso de ítem basura."""

    def detect_battle_menu(self, frame: np.ndarray) -> bool:
        """Detecta el menú principal de combate (Fight, Items, Run, Loomians)."""

    def detect_fight_cancel(self, frame: np.ndarray) -> bool:
        """Detecta el botón gris Cancel dentro del submenú de Fight."""

    def detect_fishing_return(self, frame: np.ndarray) -> bool:
        """Detecta el retorno a la vista de pesca con evidencia visual explícita."""

    def classify_encounter(self, frame: np.ndarray) -> EncounterResult:
        """Identifica nombre, forma y confianza del encuentro."""


class ControllerProtocol(Protocol):
    """Interfaz mínima requerida para acciones del bot y persistencia."""

    def is_in_run_list(self, name: str) -> bool:
        """Consulta en vivo si un Loomian figura en run_list.json."""

    def add_to_run_list(self, name: str) -> bool:
        """Agrega un Loomian a run_list.json."""

    def append_corrupt(self, loomian: str, shard: str) -> None:
        """Registra un encuentro corrupto en data/corruptos.csv."""

    def cast_rod(self) -> None:
        """Lanza la caña de pescar."""

    def move_to_dot(self, point: Point) -> None:
        """Mueve el cursor a un nodo del minijuego."""

    def click_run(self) -> None:
        """Hace clic en el botón Run."""

    def click_cancel(self) -> None:
        """Hace clic en el botón gris Cancel para volver al menú principal de combate."""

    def click_fight_and_move(self) -> None:
        """Hace clic en Fight y selecciona el movimiento contra corruptos."""

    def close_item_notice(self) -> None:
        """Cierra el aviso de ítem si corresponde."""


class AlertProtocol(Protocol):
    """Interfaz mínima para disparo de alerta sonora."""

    def trigger(self, reason: str = "") -> None:
        """Dispara la alerta sonora."""


class KillSwitchProtocol(Protocol):
    """Interfaz mínima para consultar el kill-switch global."""

    def is_triggered(self) -> bool:
        """Indica si el kill-switch fue activado."""


class FishingStateMachine:
    """Implementa todos los estados, transiciones, timeouts y reglas IN_BATTLE de SPEC.md."""

    def __init__(
        self,
        capture: ScreenCaptureProtocol,
        tracker: MinigameTrackerProtocol,
        classifier: EncounterClassifierProtocol,
        controller: ControllerProtocol,
        alert: AlertProtocol,
        kill_switch: KillSwitchProtocol,
        timeouts: Mapping[str, float],
        min_confidence: float = 0.80,
        stop_after_corrupt: bool = False,
        clock: Callable[[], float] = time.monotonic,
        battle_menu_miss_limit: int = 3,
    ) -> None:
        self.capture = capture
        self.tracker = tracker
        self.classifier = classifier
        self.controller = controller
        self.alert = alert
        self.kill_switch = kill_switch
        self.timeouts: dict[str, float] = {k: float(v) for k, v in timeouts.items()}
        self.min_confidence = min_confidence
        self.stop_after_corrupt = stop_after_corrupt
        self._clock = clock
        self.battle_menu_miss_limit = max(1, int(battle_menu_miss_limit))

        self.state: State = State.IDLE_OR_CAST
        self._state_entered_at: float = self._clock()
        self._battle_menu_misses: int = 0
        self.alert_reason: str | None = None
        self.pending_run_list_candidate: str | None = None
        self.last_frame: np.ndarray | None = None
        self.last_points: list[Point] = []
        self.last_encounter: EncounterResult | None = None

    @property
    def is_terminal(self) -> bool:
        """Indica si la máquina alcanzó un estado final seguro."""
        return self.state in TERMINAL_STATES

    def _elapsed_in_state(self) -> float:
        return self._clock() - self._state_entered_at

    def _is_timed_out(self, state: State) -> bool:
        limit = self.timeouts.get(state.value)
        if limit is None:
            return False
        return self._elapsed_in_state() >= limit

    def _transition_to(self, next_state: State, reason: str = "") -> State:
        prev = self.state
        if prev != next_state:
            logger.info(
                "Transición: %s -> %s%s",
                prev.value,
                next_state.value,
                f" ({reason})" if reason else "",
            )
        self.state = next_state
        self._state_entered_at = self._clock()
        self._battle_menu_misses = 0
        self.capture.set_solving_mode(next_state == State.SOLVING_DOTS)
        return self.state

    def _stop_with_alert(self, reason: str, run_list_candidate: str | None = None) -> State:
        self.alert_reason = reason
        self.pending_run_list_candidate = run_list_candidate
        self.alert.trigger(reason=reason)
        return self._transition_to(State.STOPPED_ALERT, reason=reason)

    def step(self) -> State:
        """Ejecuta un paso de la máquina de estados y devuelve el estado resultante."""
        if self.is_terminal:
            return self.state

        if self.kill_switch.is_triggered():
            return self._transition_to(State.ABORTED, reason="kill-switch activado")

        if self.state == State.IDLE_OR_CAST:
            return self._handle_idle_or_cast()
        if self.state == State.WAITING_MINIGAME:
            return self._handle_waiting_minigame()
        if self.state == State.SOLVING_DOTS:
            return self._handle_solving_dots()
        if self.state == State.RESOLVING_OUTCOME:
            return self._handle_resolving_outcome()
        if self.state == State.IN_BATTLE:
            return self._handle_in_battle()
        if self.state == State.COOLDOWN:
            return self._handle_cooldown()

        return self._stop_with_alert(f"Estado no reconocido: {self.state}")

    def _handle_idle_or_cast(self) -> State:
        if self._is_timed_out(State.IDLE_OR_CAST):
            return self._stop_with_alert("Timeout en IDLE_OR_CAST")
        self.controller.cast_rod()
        return self._transition_to(State.WAITING_MINIGAME, reason="caña lanzada")

    def _handle_waiting_minigame(self) -> State:
        if self._is_timed_out(State.WAITING_MINIGAME):
            return self._transition_to(State.IDLE_OR_CAST, reason="timeout esperando minijuego")

        frame = self.capture.grab()
        self.last_frame = frame
        if self.tracker.is_minigame_active(frame):
            return self._transition_to(State.SOLVING_DOTS, reason="minijuego detectado")
        return self.state

    def _handle_solving_dots(self) -> State:
        if self._is_timed_out(State.SOLVING_DOTS):
            return self._transition_to(State.RESOLVING_OUTCOME, reason="timeout en SOLVING_DOTS")

        frame = self.capture.grab()
        self.last_frame = frame
        if not self.tracker.is_minigame_active(frame):
            self.last_points = []
            return self._transition_to(State.RESOLVING_OUTCOME, reason="minijuego cerrado")

        points = self.tracker.get_ordered_nodes(frame)
        self.last_points = list(points)
        for pt in points:
            if self.kill_switch.is_triggered():
                return self._transition_to(State.ABORTED, reason="kill-switch durante SOLVING_DOTS")
            self.controller.move_to_dot(pt)
        return self.state

    def _handle_resolving_outcome(self) -> State:
        if self._is_timed_out(State.RESOLVING_OUTCOME):
            return self._transition_to(
                State.COOLDOWN,
                reason="timeout en RESOLVING_OUTCOME sin cambios claros",
            )

        frame = self.capture.grab()
        self.last_frame = frame

        if self.classifier.detect_item_notice(frame):
            self.controller.close_item_notice()
            return self._transition_to(State.COOLDOWN, reason="aviso de ítem detectado")

        if self.classifier.detect_battle_menu(frame):
            return self._transition_to(State.IN_BATTLE, reason="menú de combate detectado")

        if self.classifier.detect_fight_cancel(frame):
            logger.info("Submenú de Fight detectado (Cancel) en RESOLVING_OUTCOME -> volviendo al menú principal.")
            self.controller.click_cancel()
            return self.state

        if self.classifier.detect_fishing_return(frame):
            return self._transition_to(State.COOLDOWN, reason="retorno a pesca detectado")

        return self.state

    def _handle_in_battle(self) -> State:
        if self._is_timed_out(State.IN_BATTLE):
            return self._stop_with_alert("Timeout en IN_BATTLE")

        frame = self.capture.grab()
        self.last_frame = frame

        if not self.classifier.detect_battle_menu(frame):
            if self.classifier.detect_fight_cancel(frame):
                self._battle_menu_misses = 0
                logger.info("Submenú de Fight detectado (Cancel) en IN_BATTLE -> volviendo al menú principal.")
                self.controller.click_cancel()
                return self.state
            self._battle_menu_misses += 1
            if self._battle_menu_misses < self.battle_menu_miss_limit:
                logger.debug(
                    "Menú de batalla ausente temporalmente en IN_BATTLE (%d/%d); reintentando.",
                    self._battle_menu_misses,
                    self.battle_menu_miss_limit,
                )
                return self.state
            return self._stop_with_alert("Menú de batalla no visible o ambiguo en IN_BATTLE")

        self._battle_menu_misses = 0

        result = self.classifier.classify_encounter(frame)
        self.last_encounter = result

        # Validación de seguridad: confianza baja, forma desconocida o nombre ausente
        if (
            result.form == Form.UNKNOWN
            or not result.name
            or result.confidence < self.min_confidence
        ):
            return self._stop_with_alert(
                f"Lectura ambigua o confianza baja (name={result.name!r}, "
                f"form={result.form.value}, confidence={result.confidence:.2f})"
            )

        if result.form == Form.NORMAL:
            if self.controller.is_in_run_list(result.name):
                self.controller.click_run()
                return self._transition_to(
                    State.COOLDOWN,
                    reason=f"normal '{result.name}' en run_list -> RUN",
                )
            return self._stop_with_alert(
                f"Loomian normal '{result.name}' NO está en run_list.json"
            )

        if result.form == Form.CORRUPTO:
            self.controller.click_fight_and_move()
            shard_name = result.shard or "Unknown Shard"
            self.controller.append_corrupt(result.name, shard_name)
            if self.stop_after_corrupt:
                return self._stop_with_alert(
                    f"Corrupto '{result.name}' derrotado y stop_after_corrupt=true"
                )
            return self._transition_to(
                State.COOLDOWN,
                reason=f"corrupto '{result.name}' registrado -> continuar",
            )

        if result.form in (Form.ALPHA, Form.GAMMA):
            return self._stop_with_alert(
                f"Encuentro {result.form.value.upper()}: '{result.name}'",
                run_list_candidate=result.name,
            )

        return self._stop_with_alert(f"Forma no manejada: {result.form}")

    def _handle_cooldown(self) -> State:
        if self._is_timed_out(State.COOLDOWN):
            return self._transition_to(State.IDLE_OR_CAST, reason="cooldown completado")
        return self.state
