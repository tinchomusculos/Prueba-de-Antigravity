"""Máquina de estados principal para el ciclo de pesca en Loomian Legacy."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from core.controller import RunListError
from core.types import EncounterResult, Form, Point, ROI, State
from vision.minigame_tracker import MinigameState

logger = logging.getLogger(__name__)

TERMINAL_STATES = frozenset({State.STOPPED_ALERT, State.ABORTED})


class _BattlePhase(Enum):
    """Sub-fases internas dentro de ``IN_BATTLE`` sin alterar el Enum público ``State``."""

    CLASSIFYING = "CLASSIFYING"
    WAITING_RUN_EXIT = "WAITING_RUN_EXIT"
    WAITING_CORRUPT_DROP = "WAITING_CORRUPT_DROP"


@dataclass
class _PendingEncounterHints:
    """Buffer de indicios transitorios recolectados en ``RESOLVING_OUTCOME`` e ``IN_BATTLE``."""

    corrupt_powerful: bool = False
    corrupt_appeared: bool = False
    sparkle_form: Form | None = None
    wisp_detected: bool = False
    prev_frame: np.ndarray | None = None
    clean_form_frames: int = 0

    def reset(self) -> None:
        self.corrupt_powerful = False
        self.corrupt_appeared = False
        self.sparkle_form = None
        self.wisp_detected = False
        self.prev_frame = None
        self.clean_form_frames = 0

    def has_any_signal(self) -> bool:
        return (
            self.corrupt_powerful
            or self.corrupt_appeared
            or self.sparkle_form is not None
            or self.wisp_detected
        )


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

    def analyze(self, frame: np.ndarray) -> MinigameState:
        """Analiza el frame completo del minijuego."""


class EncounterClassifierProtocol(Protocol):
    """Interfaz mínima para detectar resultados del minijuego, combates, formas y shards."""

    def detect_shard_drop(self, frame: np.ndarray) -> str | None:
        """Detecta el aviso de drop de shard y devuelve su nombre si está visible."""

    def detect_battle_menu(self, frame: np.ndarray) -> bool:
        """Detecta el menú principal de combate (Fight, Items, Run, Loomians)."""

    def detect_fight_cancel(self, frame: np.ndarray) -> bool:
        """Detecta el botón gris Cancel dentro del submenú de Fight."""

    def detect_fishing_return(self, frame: np.ndarray) -> bool:
        """Detecta el retorno a la vista de pesca o salida de combate."""

    def detect_corrupt_powerful_banner(self, frame: np.ndarray) -> bool:
        """Detecta el banner 'It looks more powerful than normal!'."""

    def detect_corrupt_appeared_banner(self, frame: np.ndarray) -> bool:
        """Detecta el prefijo 'A corrupt' en el banner de aparición."""

    def detect_corrupt_card(self, frame: np.ndarray) -> bool:
        """Detecta si la tarjeta enemiga es violeta (corrupto)."""

    def detect_entry_sparkle(self, frame: np.ndarray) -> Form | None:
        """Detecta destellos de entrada alpha o gamma."""

    def detect_wisp(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> bool:
        """Detecta movimiento de wisp alrededor del sprite."""

    def classify_name(self, frame: np.ndarray) -> tuple[str | None, float]:
        """Identifica solo el nombre de especie del Loomian."""

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
        """Cierra o avanza el aviso inferior de drop."""


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
        dot_move_delay_s: float = 0.0,
        min_bar_ratio: float = 0.05,
        max_solving_retries: int = 5,
        sleeper: Callable[[float], None] = time.sleep,
        fishing_return_confirm_s: float = 0.0,
        dark_frame_mean_threshold: float = 25.0,
        form_confirm_frames: int = 1,
        unknown_shards_dir: str | Path = "debug/unknown_shards",
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
        self.dot_move_delay_s = max(0.0, float(dot_move_delay_s))
        self.min_bar_ratio = float(min_bar_ratio)
        self.max_solving_retries = max(1, int(max_solving_retries))
        self._sleeper = sleeper
        self.fishing_return_confirm_s = max(0.0, float(fishing_return_confirm_s))
        self.dark_frame_mean_threshold = float(dark_frame_mean_threshold)
        self.form_confirm_frames = max(1, int(form_confirm_frames))
        self.unknown_shards_dir = Path(unknown_shards_dir)

        self.state: State = State.IDLE_OR_CAST
        self._state_entered_at: float = self._clock()
        self._battle_menu_misses: int = 0
        self._battle_phase: _BattlePhase = _BattlePhase.CLASSIFYING
        self._active_corrupt_loomian: str | None = None
        self._detected_shard: str | None = None
        self._unknown_shard_saved: bool = False
        self._solving_retries: int = 0
        self._fishing_return_since: float | None = None
        self._pending_hints = _PendingEncounterHints()
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

    def _apply_capture_mode(self, next_state: State) -> None:
        set_mode = getattr(self.capture, "set_capture_mode", None)
        if callable(set_mode):
            if next_state == State.SOLVING_DOTS:
                set_mode("solving")
            elif next_state == State.RESOLVING_OUTCOME:
                set_mode("resolving")
            else:
                set_mode("idle")
        else:
            self.capture.set_solving_mode(next_state == State.SOLVING_DOTS)

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
        self._battle_phase = _BattlePhase.CLASSIFYING
        self._active_corrupt_loomian = None
        self._detected_shard = None
        self._unknown_shard_saved = False
        self._solving_retries = 0
        self._fishing_return_since = None

        # El buffer se conserva de RESOLVING_OUTCOME -> IN_BATTLE; se limpia en COOLDOWN, IDLE_OR_CAST, STOPPED_ALERT y ABORTED
        if next_state in (State.COOLDOWN, State.IDLE_OR_CAST, State.STOPPED_ALERT, State.ABORTED):
            self._pending_hints.reset()

        self._apply_capture_mode(next_state)
        return self.state

    def _stop_with_alert(self, reason: str, run_list_candidate: str | None = None) -> State:
        self.alert_reason = reason
        self.pending_run_list_candidate = run_list_candidate
        self.alert.trigger(reason=reason)
        return self._transition_to(State.STOPPED_ALERT, reason=reason)

    def _record_sparkle_hint(self, frame: np.ndarray) -> None:
        sparkle = self.classifier.detect_entry_sparkle(frame)
        if sparkle is None:
            return
        if self._pending_hints.sparkle_form is None:
            self._pending_hints.sparkle_form = sparkle
        elif self._pending_hints.sparkle_form != sparkle:
            self._pending_hints.sparkle_form = Form.UNKNOWN

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
        mg_state = self.tracker.analyze(frame)
        if not mg_state.active:
            self.last_points = []
            return self._transition_to(State.RESOLVING_OUTCOME, reason="minijuego cerrado")

        if mg_state.bar_ratio is not None and mg_state.bar_ratio < self.min_bar_ratio:
            self.last_points = []
            logger.warning(
                "Barra de tiempo baja en SOLVING_DOTS (%.3f < %.3f); pasando a RESOLVING_OUTCOME.",
                mg_state.bar_ratio,
                self.min_bar_ratio,
            )
            return self._transition_to(
                State.RESOLVING_OUTCOME,
                reason="barra de tiempo insuficiente",
            )

        points = list(mg_state.ordered_nodes)
        self.last_points = points
        if not points:
            self._solving_retries += 1
            if self._solving_retries >= self.max_solving_retries:
                return self._transition_to(
                    State.RESOLVING_OUTCOME,
                    reason="límite de reintentos sin nodos en SOLVING_DOTS",
                )
            return self.state

        self._solving_retries = 0
        for idx, pt in enumerate(points):
            if self.kill_switch.is_triggered():
                return self._transition_to(State.ABORTED, reason="kill-switch durante SOLVING_DOTS")
            self.controller.move_to_dot(pt)
            if self.dot_move_delay_s > 0 and idx < len(points) - 1:
                self._sleeper(self.dot_move_delay_s)
        return self.state

    def _handle_resolving_outcome(self) -> State:
        if self._is_timed_out(State.RESOLVING_OUTCOME):
            return self._transition_to(
                State.COOLDOWN,
                reason="timeout en RESOLVING_OUTCOME sin cambios claros",
            )

        frame = self.capture.grab()
        self.last_frame = frame

        # Acumula señales transitorias de corrupto y destellos alpha/gamma antes de entrar a IN_BATTLE
        if self.classifier.detect_corrupt_powerful_banner(frame):
            self._pending_hints.corrupt_powerful = True
        if self.classifier.detect_corrupt_appeared_banner(frame):
            self._pending_hints.corrupt_appeared = True
        self._record_sparkle_hint(frame)

        if self.classifier.detect_battle_menu(frame):
            return self._transition_to(State.IN_BATTLE, reason="menú de combate detectado")

        if self.classifier.detect_fight_cancel(frame):
            self._fishing_return_since = None
            logger.info("Submenú de Fight detectado (Cancel) en RESOLVING_OUTCOME -> volviendo al menú principal.")
            self.controller.click_cancel()
            return self.state

        mean_brightness = float(np.mean(frame)) if frame.size > 0 else 0.0
        is_dark_fade = mean_brightness < self.dark_frame_mean_threshold
        if (
            not is_dark_fade
            and not self._pending_hints.has_any_signal()
            and self.classifier.detect_fishing_return(frame)
        ):
            now = self._clock()
            if self._fishing_return_since is None:
                self._fishing_return_since = now
            if now - self._fishing_return_since >= self.fishing_return_confirm_s:
                return self._transition_to(State.COOLDOWN, reason="retorno estable a pesca detectado")
        else:
            self._fishing_return_since = None

        return self.state

    def _handle_in_battle(self) -> State:
        if self._is_timed_out(State.IN_BATTLE):
            return self._stop_with_alert("Timeout en IN_BATTLE")

        frame = self.capture.grab()
        self.last_frame = frame

        if self._battle_phase == _BattlePhase.WAITING_RUN_EXIT:
            if self.classifier.detect_fishing_return(frame):
                return self._transition_to(
                    State.COOLDOWN,
                    reason="salida de combate confirmada tras RUN",
                )
            return self.state

        if self._battle_phase == _BattlePhase.WAITING_CORRUPT_DROP:
            shard = self.classifier.detect_shard_drop(frame)
            if shard is not None:
                if shard == "Unknown":
                    self._save_unknown_shard_frame(frame)
                self._detected_shard = shard
                self.controller.close_item_notice()
                if self.classifier.detect_fishing_return(frame):
                    return self._finalize_corrupt_drop()
                return self.state

            if self._detected_shard is not None and (
                self.classifier.detect_fishing_return(frame)
                or not self.classifier.detect_battle_menu(frame)
            ):
                return self._finalize_corrupt_drop()
            return self.state

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

        # Actualiza evidencias en combate (tarjeta corrupta, destellos y wisp en movimiento)
        self._record_sparkle_hint(frame)
        if self._pending_hints.prev_frame is not None:
            if self.classifier.detect_wisp(self._pending_hints.prev_frame, frame):
                self._pending_hints.wisp_detected = True
        self._pending_hints.prev_frame = frame

        banner_powerful = self._pending_hints.corrupt_powerful or self.classifier.detect_corrupt_powerful_banner(frame)
        banner_appeared = self._pending_hints.corrupt_appeared or self.classifier.detect_corrupt_appeared_banner(frame)
        purple_card = self.classifier.detect_corrupt_card(frame)
        any_corrupt_signal = banner_powerful or banner_appeared or purple_card

        # Cualquier señal de corrupto bloquea RUN; si las señales no coinciden entre sí -> STOPPED_ALERT
        if any_corrupt_signal:
            consistent_corrupt = (
                banner_powerful
                and (banner_appeared or purple_card)
                and self._pending_hints.sparkle_form is None
                and not self._pending_hints.wisp_detected
            )
            if not consistent_corrupt:
                return self._stop_with_alert(
                    "Señales de corrupto inconsistentes o parciales "
                    f"(powerful={banner_powerful}, appeared={banner_appeared}, "
                    f"purple_card={purple_card}, sparkle={self._pending_hints.sparkle_form}, "
                    f"wisp={self._pending_hints.wisp_detected})"
                )

            name, name_conf = self.classifier.classify_name(frame)
            if not name or name_conf < self.min_confidence:
                return self._stop_with_alert(
                    f"Corrupto confirmado pero especie ambigua (name={name!r}, conf={name_conf:.2f})"
                )
            self.last_encounter = EncounterResult(name=name, form=Form.CORRUPTO, confidence=name_conf)
            return self._start_corrupt_battle(name)

        # Señales de alpha / gamma detectadas en RESOLVING_OUTCOME o IN_BATTLE
        if self._pending_hints.wisp_detected or self._pending_hints.sparkle_form in (Form.ALPHA, Form.GAMMA):
            detected_form = Form.GAMMA if self._pending_hints.wisp_detected else self._pending_hints.sparkle_form
            assert detected_form is not None
            name, name_conf = self.classifier.classify_name(frame)
            if not name or name_conf < self.min_confidence:
                return self._stop_with_alert(
                    f"Encuentro {detected_form.value.upper()} con nombre desconocido o confianza baja"
                )
            self.last_encounter = EncounterResult(name=name, form=detected_form, confidence=name_conf)
            return self._stop_with_alert(
                f"Encuentro {detected_form.value.upper()}: '{name}'",
                run_list_candidate=name,
            )

        if self._pending_hints.sparkle_form == Form.UNKNOWN:
            return self._stop_with_alert("Destello ambiguo en encuentro (alpha/gamma)")

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
            self._pending_hints.clean_form_frames += 1
            if self._pending_hints.clean_form_frames < self.form_confirm_frames:
                logger.debug(
                    "Confirmando forma normal (%d/%d frames limpios sin destellos ni wisp).",
                    self._pending_hints.clean_form_frames,
                    self.form_confirm_frames,
                )
                return self.state

            try:
                in_run_list = self.controller.is_in_run_list(result.name)
            except RunListError as exc:
                return self._stop_with_alert(f"run_list.json inválido: {exc}")

            if in_run_list:
                self.controller.click_run()
                self._battle_phase = _BattlePhase.WAITING_RUN_EXIT
                logger.info(
                    "RUN ejecutado contra '%s'; esperando confirmación de salida de combate.",
                    result.name,
                )
                return self.state
            return self._stop_with_alert(
                f"Loomian normal '{result.name}' NO está en run_list.json"
            )

        if result.form == Form.CORRUPTO:
            return self._start_corrupt_battle(result.name)

        if result.form in (Form.ALPHA, Form.GAMMA):
            return self._stop_with_alert(
                f"Encuentro {result.form.value.upper()}: '{result.name}'",
                run_list_candidate=result.name,
            )

        return self._stop_with_alert(f"Forma no manejada: {result.form}")

    def _start_corrupt_battle(self, loomian_name: str) -> State:
        self.controller.click_fight_and_move()
        self._active_corrupt_loomian = loomian_name
        self._detected_shard = None
        self._unknown_shard_saved = False
        self._battle_phase = _BattlePhase.WAITING_CORRUPT_DROP
        logger.info(
            "Ataque ejecutado contra corrupto '%s'; esperando drop del shard y fin de combate.",
            loomian_name,
        )
        return self.state

    def _save_unknown_shard_frame(self, frame: np.ndarray) -> None:
        if self._unknown_shard_saved:
            return
        self.unknown_shards_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(round(self._clock() * 1000))
        dest = self.unknown_shards_dir / f"unknown_shard_{stamp}.png"
        cv2.imwrite(str(dest), frame)
        self._unknown_shard_saved = True
        logger.warning("Drop de shard confirmado pero tipo no legible; guardado en %s", dest)

    def _finalize_corrupt_drop(self) -> State:
        """Registra en CSV el corrupto con su shard real (o 'Unknown') y decide siguiente transición."""
        loomian = self._active_corrupt_loomian or "Unknown"
        shard_name = self._detected_shard or "Unknown"
        self.controller.append_corrupt(loomian, shard_name)
        if self.stop_after_corrupt:
            return self._stop_with_alert(
                f"Corrupto '{loomian}' derrotado ({shard_name}) y stop_after_corrupt=true"
            )
        return self._transition_to(
            State.COOLDOWN,
            reason=f"corrupto '{loomian}' ({shard_name}) registrado -> continuar",
        )

    def _handle_cooldown(self) -> State:
        if self._is_timed_out(State.COOLDOWN):
            return self._transition_to(State.IDLE_OR_CAST, reason="cooldown completado")
        return self.state
