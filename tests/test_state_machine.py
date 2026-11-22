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
from main import prompt_add_to_run_list
from vision.encounter_classifier import EncounterClassifier

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
        self.frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    def set_solving_mode(self, solving: bool) -> None:
        self.solving_mode = solving

    def grab(self, roi: ROI | None = None) -> np.ndarray:
        _ = roi
        return self.frame


class FakeTracker:
    def __init__(self, active: bool = False, nodes: list[Point] | None = None) -> None:
        self.active = active
        self.nodes = nodes or []

    def is_minigame_active(self, frame: np.ndarray) -> bool:
        _ = frame
        return self.active

    def get_ordered_nodes(self, frame: np.ndarray) -> list[Point]:
        _ = frame
        return list(self.nodes)


class FakeClassifier:
    def __init__(
        self,
        item_notice: bool = False,
        battle_menu: bool = True,
        fight_cancel: bool = False,
        fishing_return: bool = False,
        result: EncounterResult | None = None,
    ) -> None:
        self.item_notice = item_notice
        self.battle_menu = battle_menu
        self.fight_cancel = fight_cancel
        self.fishing_return = fishing_return
        self.result = result or EncounterResult(name=None, form=Form.UNKNOWN, confidence=0.0)

    def detect_item_notice(self, frame: np.ndarray) -> bool:
        _ = frame
        return self.item_notice

    def detect_battle_menu(self, frame: np.ndarray) -> bool:
        _ = frame
        return self.battle_menu

    def detect_fight_cancel(self, frame: np.ndarray) -> bool:
        _ = frame
        return self.fight_cancel

    def detect_fishing_return(self, frame: np.ndarray) -> bool:
        _ = frame
        return self.fishing_return

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


def test_in_battle_normal_in_run_list_runs_and_enters_cooldown(tmp_path: Path) -> None:
    classifier = FakeClassifier(
        battle_menu=True,
        result=EncounterResult(name="Goppie", form=Form.NORMAL, confidence=0.95),
    )
    sm, _, _, driver, alert, _ = _build_machine(tmp_path, classifier)

    next_state = sm.step()

    assert next_state == State.COOLDOWN
    assert alert.reasons == []
    assert driver.recorded_actions == [("click", (UI_POINTS["run_button"], "left"))]


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
def test_in_battle_corrupt_fights_logs_csv_and_respects_flag(
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
            shard="Energy Shard",
        ),
    )
    sm, _, _, driver, alert, controller = _build_machine(
        tmp_path,
        classifier,
        stop_after_corrupt=stop_after_corrupt,
    )

    next_state = sm.step()

    assert next_state == expected_state
    assert driver.recorded_actions == [
        ("click", (UI_POINTS["fight_button"], "left")),
        ("click", (UI_POINTS["corrupt_move_slot"], "left")),
    ]
    assert len(alert.reasons) == (1 if stop_after_corrupt else 0)

    with controller.corruptos_csv_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    assert len(rows) == 2
    assert rows[0] == ["timestamp", "loomian", "shard"]
    assert rows[1][1:] == ["Goppie", "Energy Shard"]


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
    classifier = FakeClassifier(item_notice=False, battle_menu=False)
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
        item_notice=False,
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
        item_notice=False,
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
    assert sm.step() == State.COOLDOWN
    assert driver.recorded_actions[-1] == ("click", (UI_POINTS["run_button"], "left"))


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


