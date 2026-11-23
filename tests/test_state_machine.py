"""Tests para core/state_machine.py: reglas IN_BATTLE, timeouts y clasificación."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from core.controller import BotController, DryRunInputDriver
from core.state_machine import FishingStateMachine
from core.types import EncounterResult, Form, Point, ROI, State
from main import (
    ConfigError,
    load_settings,
    main as cli_main,
    prompt_add_to_run_list,
    validate_run_list_file,
    validate_settings,
)
from vision.encounter_classifier import EncounterClassifier
from vision.minigame_tracker import MinigameState

DEFAULT_TIMEOUTS = {
    "IDLE_OR_CAST": 5.0,
    "WAITING_MINIGAME": 20.0,
    "SOLVING_DOTS": 12.0,
    "RESOLVING_OUTCOME": 10.0,
    "IN_BATTLE": 30.0,
    "COOLDOWN": 2.0,
}

UI_POINTS = {
    "cast_button": Point(100, 100),
    "fight_button": Point(200, 200),
    "run_button": Point(300, 300),
    "cancel_button": Point(300, 300),
    "corrupt_move_slot": Point(250, 250),
    "item_notice_close": Point(400, 400),
}


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeCapture:
    def __init__(self) -> None:
        self.solving_mode = False
        self.capture_mode = "idle"
        self.frame = np.full((720, 1280, 3), 80, dtype=np.uint8)

    def set_solving_mode(self, solving: bool) -> None:
        self.solving_mode = solving
        self.capture_mode = "solving" if solving else "idle"

    def set_capture_mode(self, mode: str) -> None:
        self.capture_mode = mode
        self.solving_mode = mode == "solving"

    def grab(self, roi: ROI | None = None) -> np.ndarray:
        _ = roi
        return self.frame


class FakeTracker:
    def __init__(
        self,
        active: bool = False,
        nodes: list[Point] | None = None,
        bar_ratio: float | None = 0.80,
    ) -> None:
        self.active = active
        self.nodes = nodes or []
        self.bar_ratio = bar_ratio

    def is_minigame_active(self, frame: np.ndarray) -> bool:
        _ = frame
        return self.active

    def get_ordered_nodes(self, frame: np.ndarray) -> list[Point]:
        _ = frame
        return list(self.nodes)

    def analyze(self, frame: np.ndarray) -> MinigameState:
        _ = frame
        if not self.active:
            return MinigameState(active=False)
        return MinigameState(
            active=True,
            bar_ratio=self.bar_ratio,
            start_node=self.nodes[0] if self.nodes else None,
            ordered_nodes=tuple(self.nodes),
        )


class FakeClassifier:
    def __init__(
        self,
        battle_menu: bool = True,
        fight_cancel: bool = False,
        fishing_return: bool = False,
        shard_drop: str | None = None,
        corrupt_powerful: bool = False,
        corrupt_appeared: bool = False,
        corrupt_card: bool = False,
        entry_sparkle: Form | None = None,
        wisp_detected: bool = False,
        name_override: tuple[str | None, float] | None = None,
        result: EncounterResult | None = None,
    ) -> None:
        self.battle_menu = battle_menu
        self.fight_cancel = fight_cancel
        self.fishing_return = fishing_return
        self.shard_drop = shard_drop
        self.corrupt_powerful = corrupt_powerful
        self.corrupt_appeared = corrupt_appeared
        self.corrupt_card = corrupt_card
        self.entry_sparkle = entry_sparkle
        self.wisp_detected = wisp_detected
        self.name_override = name_override
        self.result = result or EncounterResult(name=None, form=Form.UNKNOWN, confidence=0.0)

    def detect_shard_drop(self, frame: np.ndarray) -> str | None:
        _ = frame
        return self.shard_drop

    def detect_battle_menu(self, frame: np.ndarray) -> bool:
        _ = frame
        return self.battle_menu

    def detect_fight_cancel(self, frame: np.ndarray) -> bool:
        _ = frame
        return self.fight_cancel

    def detect_fishing_return(self, frame: np.ndarray) -> bool:
        _ = frame
        return self.fishing_return

    def detect_corrupt_powerful_banner(self, frame: np.ndarray) -> bool:
        _ = frame
        return self.corrupt_powerful

    def detect_corrupt_appeared_banner(self, frame: np.ndarray) -> bool:
        _ = frame
        return self.corrupt_appeared

    def detect_corrupt_card(self, frame: np.ndarray) -> bool:
        _ = frame
        return self.corrupt_card

    def detect_entry_sparkle(self, frame: np.ndarray) -> Form | None:
        _ = frame
        return self.entry_sparkle

    def detect_wisp(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> bool:
        _ = prev_frame, curr_frame
        return self.wisp_detected

    def classify_name(self, frame: np.ndarray) -> tuple[str | None, float]:
        _ = frame
        if self.name_override is not None:
            return self.name_override
        return self.result.name, self.result.confidence

    def classify_encounter(self, frame: np.ndarray) -> EncounterResult:
        _ = frame
        return self.result


class FakeAlert:
    def __init__(self) -> None:
        self.reasons: list[str] = []

    def trigger(self, reason: str = "") -> None:
        self.reasons.append(reason)


class FakeKillSwitch:
    def __init__(self, triggered: bool = False) -> None:
        self._triggered = triggered

    def is_triggered(self) -> bool:
        return self._triggered

    def trigger(self) -> None:
        self._triggered = True


def _build_machine(
    tmp_path: Path,
    classifier: FakeClassifier,
    tracker: FakeTracker | None = None,
    stop_after_corrupt: bool = False,
    run_list: list[str] | None = None,
    initial_state: State = State.IN_BATTLE,
) -> tuple[FishingStateMachine, FakeClock, FakeCapture, DryRunInputDriver, FakeAlert, BotController]:
    run_list_file = tmp_path / "run_list.json"
    run_list_file.write_text(
        json.dumps({"run_list": run_list if run_list is not None else ["Goppie", "Cavenish"]}, indent=2),
        encoding="utf-8",
    )
    csv_file = tmp_path / "corruptos.csv"
    driver = DryRunInputDriver()
    controller = BotController(
        run_list_path=run_list_file,
        corruptos_csv_path=csv_file,
        ui_points=UI_POINTS,
        input_driver=driver,
        dry_run=True,
    )
    clock = FakeClock()
    capture = FakeCapture()
    alert = FakeAlert()
    sm = FishingStateMachine(
        capture=capture,
        tracker=tracker or FakeTracker(),
        classifier=classifier,
        controller=controller,
        alert=alert,
        kill_switch=FakeKillSwitch(),
        timeouts=DEFAULT_TIMEOUTS,
        min_confidence=0.80,
        stop_after_corrupt=stop_after_corrupt,
        clock=clock,
    )
    sm.state = initial_state
    return sm, clock, capture, driver, alert, controller


def test_in_battle_normal_in_run_list_runs_and_waits_for_exit(tmp_path: Path) -> None:
    classifier = FakeClassifier(
        battle_menu=True,
        fishing_return=False,
        result=EncounterResult(name="Goppie", form=Form.NORMAL, confidence=0.95),
    )
    sm, _, _, driver, alert, _ = _build_machine(tmp_path, classifier)

    # Paso 1: hace clic en Run y espera confirmación de salida sin pasar aún a COOLDOWN
    assert sm.step() == State.IN_BATTLE
    assert driver.recorded_actions == [("click", (UI_POINTS["run_button"], "left"))]
    assert alert.reasons == []

    # Paso 2: todavía saliendo del combate
    classifier.battle_menu = False
    assert sm.step() == State.IN_BATTLE

    # Paso 3: se detecta salida del combate ("You got away safely!" / vista de pesca) -> COOLDOWN
    classifier.fishing_return = True
    assert sm.step() == State.COOLDOWN
    assert alert.reasons == []


def test_in_battle_run_timeout_without_exit_confirmation_stops(tmp_path: Path) -> None:
    classifier = FakeClassifier(
        battle_menu=True,
        fishing_return=False,
        result=EncounterResult(name="Goppie", form=Form.NORMAL, confidence=0.95),
    )
    sm, clock, _, _, alert, _ = _build_machine(tmp_path, classifier)

    assert sm.step() == State.IN_BATTLE
    clock.advance(DEFAULT_TIMEOUTS["IN_BATTLE"])
    assert sm.step() == State.STOPPED_ALERT
    assert len(alert.reasons) == 1


def test_in_battle_invalid_run_list_triggers_stopped_alert(tmp_path: Path) -> None:
    classifier = FakeClassifier(
        battle_menu=True,
        result=EncounterResult(name="Goppie", form=Form.NORMAL, confidence=0.95),
    )
    sm, _, _, driver, alert, controller = _build_machine(tmp_path, classifier)
    controller.run_list_path.write_text('{\n  "run_list": ["Goppie",]\n}\n', encoding="utf-8")

    next_state = sm.step()

    assert next_state == State.STOPPED_ALERT
    assert driver.recorded_actions == []
    assert len(alert.reasons) == 1
    assert "run_list.json inválido" in alert.reasons[0]


def test_in_battle_normal_not_in_run_list_alerts_and_stops(tmp_path: Path) -> None:
    classifier = FakeClassifier(
        battle_menu=True,
        result=EncounterResult(name="Duskit", form=Form.NORMAL, confidence=0.95),
    )
    sm, _, _, driver, alert, _ = _build_machine(tmp_path, classifier)

    next_state = sm.step()

    assert next_state == State.STOPPED_ALERT
    assert len(alert.reasons) == 1
    assert driver.recorded_actions == []


@pytest.mark.parametrize("stop_after_corrupt,expected_state", [(False, State.COOLDOWN), (True, State.STOPPED_ALERT)])
def test_in_battle_corrupt_waits_for_drop_and_logs_real_shard(
    tmp_path: Path,
    stop_after_corrupt: bool,
    expected_state: State,
) -> None:
    classifier = FakeClassifier(
        battle_menu=True,
        result=EncounterResult(
            name="Goppie",
            form=Form.CORRUPTO,
            confidence=0.92,
        ),
    )
    sm, _, _, driver, alert, controller = _build_machine(
        tmp_path,
        classifier,
        stop_after_corrupt=stop_after_corrupt,
    )

    # Paso 1: ataca; NO registra CSV ni lanza la caña mientras combate sigue activo
    assert sm.step() == State.IN_BATTLE
    assert driver.recorded_actions == [
        ("click", (UI_POINTS["fight_button"], "left")),
        ("click", (UI_POINTS["corrupt_move_slot"], "left")),
    ]
    assert not controller.corruptos_csv_path.exists()

    # Paso 2: animación del ataque (combate sigue activo, sin drop aún) -> sigue en IN_BATTLE
    classifier.battle_menu = False
    assert sm.step() == State.IN_BATTLE
    assert not controller.corruptos_csv_path.exists()

    # Paso 3: aparece el aviso de drop del shard -> lo cierra sin cambiar todavía a IDLE_OR_CAST
    classifier.shard_drop = "Attack Shard"
    assert sm.step() == State.IN_BATTLE
    assert driver.recorded_actions[-1] == ("click", (UI_POINTS["item_notice_close"], "left"))

    # Paso 4: se cierra el drop y vuelve a vista de pesca -> escribe CSV con shard real
    classifier.shard_drop = None
    classifier.fishing_return = True
    next_state = sm.step()

    assert next_state == expected_state
    assert len(alert.reasons) == (1 if stop_after_corrupt else 0)

    with controller.corruptos_csv_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    assert len(rows) == 2
    assert rows[0] == ["timestamp", "loomian", "shard"]
    assert rows[1][1:] == ["Goppie", "Attack Shard"]


def test_in_battle_corrupt_timeout_without_drop_stops_and_skips_csv(tmp_path: Path) -> None:
    classifier = FakeClassifier(
        battle_menu=True,
        result=EncounterResult(name="Goppie", form=Form.CORRUPTO, confidence=0.92),
    )
    sm, clock, _, _, alert, controller = _build_machine(tmp_path, classifier)

    assert sm.step() == State.IN_BATTLE
    clock.advance(DEFAULT_TIMEOUTS["IN_BATTLE"])
    assert sm.step() == State.STOPPED_ALERT
    assert len(alert.reasons) == 1
    assert not controller.corruptos_csv_path.exists()


@pytest.mark.parametrize("form", [Form.ALPHA, Form.GAMMA])
def test_in_battle_alpha_and_gamma_alert_stop_and_offer_run_list(
    tmp_path: Path,
    form: Form,
) -> None:
    classifier = FakeClassifier(
        battle_menu=True,
        result=EncounterResult(name="Pyramind", form=form, confidence=0.98),
    )
    sm, _, _, driver, alert, controller = _build_machine(tmp_path, classifier)

    next_state = sm.step()

    assert next_state == State.STOPPED_ALERT
    assert len(alert.reasons) == 1
    assert driver.recorded_actions == []
    assert sm.pending_run_list_candidate == "Pyramind"

    # Recién después de detenerse el loop, se ofrece agregar a run_list.json
    added = prompt_add_to_run_list(sm.pending_run_list_candidate, controller, input_fn=lambda _: "s")
    assert added is True
    assert controller.is_in_run_list("Pyramind") is True


def test_in_battle_low_confidence_alerts_and_stops(tmp_path: Path) -> None:
    classifier = FakeClassifier(
        battle_menu=True,
        result=EncounterResult(name="Goppie", form=Form.NORMAL, confidence=0.50),
    )
    sm, _, _, driver, alert, _ = _build_machine(tmp_path, classifier)

    next_state = sm.step()

    assert next_state == State.STOPPED_ALERT
    assert len(alert.reasons) == 1
    assert driver.recorded_actions == []


def test_state_timeouts(tmp_path: Path) -> None:
    classifier = FakeClassifier(battle_menu=False)
    tracker = FakeTracker(active=False)
    sm, clock, capture, _, alert, _ = _build_machine(
        tmp_path,
        classifier,
        tracker=tracker,
        initial_state=State.IDLE_OR_CAST,
    )

    # IDLE_OR_CAST -> WAITING_MINIGAME
    assert sm.step() == State.WAITING_MINIGAME
    # Timeout en WAITING_MINIGAME -> vuelve a IDLE_OR_CAST
    clock.advance(DEFAULT_TIMEOUTS["WAITING_MINIGAME"])
    assert sm.step() == State.IDLE_OR_CAST

    # Timeout en SOLVING_DOTS -> RESOLVING_OUTCOME y vuelve a modo idle_fps
    sm._transition_to(State.SOLVING_DOTS)
    assert capture.solving_mode is True
    clock.advance(DEFAULT_TIMEOUTS["SOLVING_DOTS"])
    assert sm.step() == State.RESOLVING_OUTCOME
    assert capture.solving_mode is False

    # Timeout en RESOLVING_OUTCOME -> COOLDOWN
    clock.advance(DEFAULT_TIMEOUTS["RESOLVING_OUTCOME"])
    assert sm.step() == State.COOLDOWN

    # Timeout en COOLDOWN -> IDLE_OR_CAST
    clock.advance(DEFAULT_TIMEOUTS["COOLDOWN"])
    assert sm.step() == State.IDLE_OR_CAST

    # Timeout en IN_BATTLE -> STOPPED_ALERT con alerta
    sm._transition_to(State.IN_BATTLE)
    clock.advance(DEFAULT_TIMEOUTS["IN_BATTLE"])
    assert sm.step() == State.STOPPED_ALERT
    assert len(alert.reasons) == 1


def test_resolving_outcome_returns_to_cooldown_when_fishing_view_is_detected(
    tmp_path: Path,
) -> None:
    classifier = FakeClassifier(
        battle_menu=False,
        fishing_return=True,
    )
    sm, _, _, _, _, _ = _build_machine(
        tmp_path,
        classifier,
        initial_state=State.RESOLVING_OUTCOME,
    )

    assert sm.step() == State.COOLDOWN


def test_classifier_rejects_mismatched_window_size(tmp_path: Path) -> None:
    classifier = EncounterClassifier(
        templates_dir=tmp_path,
        window_size=(1280, 720),
        rois={"loomian_name": ROI(0, 0, 100, 50)},
    )
    wrong_frame = np.zeros((600, 800, 3), dtype=np.uint8)

    result = classifier.classify_encounter(wrong_frame)
    assert result.form == Form.UNKNOWN
    assert result.name is None
    assert result.confidence == 0.0


def test_fight_cancel_button_returns_to_main_menu_from_in_battle_and_resolving(
    tmp_path: Path,
) -> None:
    classifier = FakeClassifier(
        battle_menu=False,
        fight_cancel=True,
        result=EncounterResult(name="Goppie", form=Form.NORMAL, confidence=0.95),
    )
    sm, _, _, driver, alert, _ = _build_machine(
        tmp_path,
        classifier,
        initial_state=State.RESOLVING_OUTCOME,
    )

    # En RESOLVING_OUTCOME con submenú Fight abierto: presiona Cancel para volver al menú principal
    assert sm.step() == State.RESOLVING_OUTCOME
    assert driver.recorded_actions == [("click", (UI_POINTS["cancel_button"], "left"))]

    # Al aparecer el menú principal, transiciona a IN_BATTLE
    classifier.fight_cancel = False
    classifier.battle_menu = True
    assert sm.step() == State.IN_BATTLE

    # Si estando en IN_BATTLE vuelve a estar abierto el submenú Fight (Cancel), presiona Cancel
    classifier.battle_menu = False
    classifier.fight_cancel = True
    assert sm.step() == State.IN_BATTLE
    assert alert.reasons == []
    assert driver.recorded_actions[-1] == ("click", (UI_POINTS["cancel_button"], "left"))

    # Ya en menú principal, ejecuta la regla normal (Goppie en run_list -> RUN)
    classifier.fight_cancel = False
    classifier.battle_menu = True
    assert sm.step() == State.IN_BATTLE
    assert driver.recorded_actions[-1] == ("click", (UI_POINTS["run_button"], "left"))

    # Confirma salida del combate antes de pasar a COOLDOWN
    classifier.battle_menu = False
    classifier.fishing_return = True
    assert sm.step() == State.COOLDOWN


def test_in_battle_tolerates_transient_menu_flicker_then_alerts(tmp_path: Path) -> None:
    classifier = FakeClassifier(
        battle_menu=False,
        fight_cancel=False,
        result=EncounterResult(name="Goppie", form=Form.NORMAL, confidence=0.95),
    )
    sm, _, _, _, alert, _ = _build_machine(tmp_path, classifier, initial_state=State.IN_BATTLE)

    # 2 frames transitorios sin menú no deben disparar STOPPED_ALERT inmediatamente
    assert sm.step() == State.IN_BATTLE
    assert sm.step() == State.IN_BATTLE
    assert alert.reasons == []

    # Al 3.er frame consecutivo ausente sí detiene de forma segura con alerta
    assert sm.step() == State.STOPPED_ALERT
    assert len(alert.reasons) == 1


def test_solving_dots_checks_bar_ratio_delay_and_retry_limit(tmp_path: Path) -> None:
    sleep_calls: list[float] = []
    tracker = FakeTracker(
        active=True,
        nodes=[Point(10, 10), Point(20, 20), Point(30, 30)],
        bar_ratio=0.50,
    )
    classifier = FakeClassifier(battle_menu=False)
    sm, _, _, driver, _, _ = _build_machine(
        tmp_path,
        classifier,
        tracker=tracker,
        initial_state=State.SOLVING_DOTS,
    )
    sm.dot_move_delay_s = 0.04
    sm.min_bar_ratio = 0.15
    sm.max_solving_retries = 2
    sm._sleeper = sleep_calls.append

    # Con barra suficiente y nodos, mueve por los 3 puntos con delay entre ellos
    assert sm.step() == State.SOLVING_DOTS
    assert len(driver.recorded_actions) == 3
    assert sleep_calls == [0.04, 0.04]

    # Sin nodos durante max_solving_retries (2) frames -> pasa a RESOLVING_OUTCOME
    tracker.nodes = []
    assert sm.step() == State.SOLVING_DOTS
    assert sm.step() == State.RESOLVING_OUTCOME

    # Si bar_ratio < min_bar_ratio -> aborta intento y pasa directo a RESOLVING_OUTCOME
    sm._transition_to(State.SOLVING_DOTS)
    tracker.nodes = [Point(10, 10)]
    tracker.bar_ratio = 0.05
    assert sm.step() == State.RESOLVING_OUTCOME


def test_startup_validation_catches_invalid_settings_and_run_list(tmp_path: Path) -> None:
    bad_json_path = tmp_path / "bad_settings.json"
    bad_json_path.write_text('{"window_size": }', encoding="utf-8")
    with pytest.raises(ConfigError, match="no es JSON válido"):
        load_settings(bad_json_path)

    settings = {
        "window_size": {"width": 1000, "height": 800},
        "rois": {
            "minigame": {"x": 900, "y": 100, "width": 200, "height": 200},
        },
        "ui_points": {"cast_button": {"x": 1500, "y": 100}},
        "timeouts": {"IDLE_OR_CAST": -1},
        "capture_rate_limit": {"idle_fps": 0, "solving_dots_fps": 30},
        "kill_switch_key": "",
        "stop_after_corrupt": "false",
        "template_match_threshold": 1.5,
        "min_confidence": 0.8,
        "templates_dir": "assets/templates",
        "paths": {"run_list": "config/run_list.json", "corruptos_csv": "data/corruptos.csv"},
    }
    errors = validate_settings(settings)
    assert any("rois.minigame" in e and "fuera de window_size" in e for e in errors)
    assert any("ui_points.cast_button" in e and "fuera de window_size" in e for e in errors)
    assert any("timeouts.IDLE_OR_CAST" in e for e in errors)
    assert any("stop_after_corrupt" in e for e in errors)

    bad_run_list = tmp_path / "run_list.json"
    bad_run_list.write_text('{"run_list": ["Goppie",]}', encoding="utf-8")
    assert validate_run_list_file(bad_run_list)

    # CLI aborta con código 1 ante settings.json inválido
    assert cli_main(["--settings", str(bad_json_path)]) == 1

    # El settings.json y run_list.json reales del repositorio deben pasar la validación limpios
    assert validate_settings(load_settings("config/settings.json")) == []
    assert validate_run_list_file("config/run_list.json") == []


def test_resolving_outcome_fps_mode_confirm_window_and_dark_frame_rejection(tmp_path: Path) -> None:
    classifier = FakeClassifier(battle_menu=False, fishing_return=True)
    sm, clock, capture, _, _, _ = _build_machine(
        tmp_path,
        classifier,
        initial_state=State.SOLVING_DOTS,
    )
    sm.fishing_return_confirm_s = 2.0
    sm.dark_frame_mean_threshold = 25.0

    # SOLVING_DOTS -> RESOLVING_OUTCOME cambia a modo resolving
    assert sm.step() == State.RESOLVING_OUTCOME
    assert capture.capture_mode == "resolving"

    # Frame oscuro (fade a combate): aunque fishing_return=True, NO inicia ni pasa a COOLDOWN
    capture.frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert sm.step() == State.RESOLVING_OUTCOME
    clock.advance(2.5)
    assert sm.step() == State.RESOLVING_OUTCOME

    # Frame iluminado con fishing_return=True: espera ventana de 2.0s antes de pasar a COOLDOWN
    capture.frame = np.full((720, 1280, 3), 90, dtype=np.uint8)
    assert sm.step() == State.RESOLVING_OUTCOME
    clock.advance(1.5)
    assert sm.step() == State.RESOLVING_OUTCOME
    clock.advance(0.6)
    assert sm.step() == State.COOLDOWN
    assert capture.capture_mode == "idle"


def test_corrupt_buffer_bridges_resolving_to_in_battle_and_resets(tmp_path: Path) -> None:
    classifier = FakeClassifier(
        battle_menu=False,
        corrupt_appeared=True,
        corrupt_powerful=False,
        name_override=("Goppie", 0.95),
    )
    sm, _, _, driver, alert, _ = _build_machine(
        tmp_path,
        classifier,
        initial_state=State.RESOLVING_OUTCOME,
    )

    # Banner 1 en RESOLVING_OUTCOME ("A corrupt ...")
    assert sm.step() == State.RESOLVING_OUTCOME
    # Banner 2 en RESOLVING_OUTCOME ("It looks more powerful than normal!")
    classifier.corrupt_appeared = False
    classifier.corrupt_powerful = True
    assert sm.step() == State.RESOLVING_OUTCOME

    # Aparece el menú de batalla sin banner ni tarjeta violeta extra; el buffer confirma corrupto -> Fight
    classifier.corrupt_powerful = False
    classifier.battle_menu = True
    assert sm.step() == State.IN_BATTLE
    assert sm.step() == State.IN_BATTLE
    assert driver.recorded_actions == [
        ("click", (UI_POINTS["fight_button"], "left")),
        ("click", (UI_POINTS["corrupt_move_slot"], "left")),
    ]

    # Confirma drop y pasa a COOLDOWN -> buffer debe quedar limpio
    classifier.shard_drop = "Speed Shard"
    assert sm.step() == State.IN_BATTLE
    classifier.shard_drop = None
    classifier.fishing_return = True
    assert sm.step() == State.COOLDOWN
    assert sm._pending_hints.has_any_signal() is False
    assert alert.reasons == []


def test_inconsistent_corrupt_signals_block_run_and_stop(tmp_path: Path) -> None:
    # Solo tarjeta violeta sin banners, aunque Goppie está en run_list -> STOPPED_ALERT, NUNCA Run
    classifier = FakeClassifier(
        battle_menu=True,
        corrupt_card=True,
        corrupt_powerful=False,
        corrupt_appeared=False,
        result=EncounterResult(name="Goppie", form=Form.NORMAL, confidence=0.98),
    )
    sm, _, _, driver, alert, _ = _build_machine(tmp_path, classifier, initial_state=State.IN_BATTLE)

    assert sm.step() == State.STOPPED_ALERT
    assert len(alert.reasons) == 1
    assert "inconsistentes" in alert.reasons[0]
    assert driver.recorded_actions == []
    assert sm._pending_hints.has_any_signal() is False


def test_unknown_shard_saves_frame_logs_unknown_and_continues(tmp_path: Path) -> None:
    unknown_dir = tmp_path / "unknown_shards"
    classifier = FakeClassifier(
        battle_menu=True,
        corrupt_powerful=True,
        corrupt_card=True,
        name_override=("Goppie", 0.96),
    )
    sm, _, _, driver, alert, controller = _build_machine(
        tmp_path,
        classifier,
        initial_state=State.IN_BATTLE,
    )
    sm.unknown_shards_dir = unknown_dir

    # Ataque contra corrupto
    assert sm.step() == State.IN_BATTLE
    # Drop confirmado pero shard ilegible -> "Unknown"
    classifier.battle_menu = False
    classifier.shard_drop = "Unknown"
    assert sm.step() == State.IN_BATTLE
    saved = list(unknown_dir.glob("unknown_shard_*.png"))
    assert len(saved) == 1
    assert driver.recorded_actions[-1] == ("click", (UI_POINTS["item_notice_close"], "left"))

    # Al cerrar el aviso y volver a pesca, escribe CSV con "Unknown" y sigue a COOLDOWN (no frena)
    classifier.shard_drop = None
    classifier.fishing_return = True
    assert sm.step() == State.COOLDOWN
    assert alert.reasons == []
    rows = list(csv.reader(controller.corruptos_csv_path.open("r", encoding="utf-8", newline="")))
    assert rows[1][1:] == ["Goppie", "Unknown"]


def test_normal_confirm_window_and_wisp_gamma_detection(tmp_path: Path) -> None:
    classifier = FakeClassifier(
        battle_menu=True,
        result=EncounterResult(name="Goppie", form=Form.NORMAL, confidence=0.95),
    )
    sm, _, _, driver, alert, _ = _build_machine(tmp_path, classifier, initial_state=State.IN_BATTLE)
    sm.form_confirm_frames = 3

    # Frames 1 y 2 limpios: todavía confirma ventana antes de ejecutar Run
    assert sm.step() == State.IN_BATTLE
    assert driver.recorded_actions == []
    assert sm.step() == State.IN_BATTLE
    assert driver.recorded_actions == []

    # Si en el frame 3 aparece un wisp -> detecta GAMMA y frena con alerta (no hace Run)
    classifier.wisp_detected = True
    assert sm.step() == State.STOPPED_ALERT
    assert len(alert.reasons) == 1
    assert "GAMMA" in alert.reasons[0]
    assert driver.recorded_actions == []




