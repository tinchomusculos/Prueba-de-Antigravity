"""Tests para el adaptador de kill-switch y su integración con la máquina de estados."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.controller import BotController, DryRunInputDriver
from core.state_machine import FishingStateMachine
from core.types import Point, ROI, State
from main import KillSwitchRegistrationError, setup_kill_switch
from vision.minigame_tracker import MinigameState


class FakeKillSwitch:
    def __init__(self, fail_on_register: bool = False) -> None:
        self.fail_on_register = fail_on_register
        self.registered = False
        self._triggered = False

    def register(self) -> None:
        if self.fail_on_register:
            raise KillSwitchRegistrationError("permiso denegado")
        self.registered = True

    def unregister(self) -> None:
        self.registered = False

    def trigger(self) -> None:
        self._triggered = True

    def is_triggered(self) -> bool:
        return self._triggered


class DummyCapture:
    def set_solving_mode(self, solving: bool) -> None:
        _ = solving

    def grab(self, roi: ROI | None = None) -> np.ndarray:
        _ = roi
        return np.zeros((720, 1280, 3), dtype=np.uint8)


class DummyTracker:
    def is_minigame_active(self, frame: np.ndarray) -> bool:
        _ = frame
        return True

    def get_ordered_nodes(self, frame: np.ndarray) -> list[Point]:
        _ = frame
        return [Point(10, 10), Point(20, 20)]

    def analyze(self, frame: np.ndarray) -> MinigameState:
        _ = frame
        return MinigameState(
            active=True,
            bar_ratio=0.80,
            start_node=Point(10, 10),
            ordered_nodes=(Point(10, 10), Point(20, 20)),
        )


class DummyClassifier:
    def detect_shard_drop(self, frame: np.ndarray) -> str | None:
        _ = frame
        return None

    def detect_battle_menu(self, frame: np.ndarray) -> bool:
        _ = frame
        return False

    def detect_fight_cancel(self, frame: np.ndarray) -> bool:
        _ = frame
        return False

    def detect_fishing_return(self, frame: np.ndarray) -> bool:
        _ = frame
        return False

    def detect_corrupt_powerful_banner(self, frame: np.ndarray) -> bool:
        _ = frame
        return False

    def detect_corrupt_appeared_banner(self, frame: np.ndarray) -> bool:
        _ = frame
        return False

    def detect_corrupt_card(self, frame: np.ndarray) -> bool:
        _ = frame
        return False

    def detect_entry_sparkle(self, frame: np.ndarray):
        _ = frame
        return None

    def detect_wisp(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> bool:
        _ = prev_frame, curr_frame
        return False

    def classify_name(self, frame: np.ndarray) -> tuple[str | None, float]:
        _ = frame
        return None, 0.0

    def classify_encounter(self, frame: np.ndarray):  # pragma: no cover
        raise AssertionError("No debería invocarse cuando el kill-switch está activo")


class DummyAlert:
    def trigger(self, reason: str = "") -> None:
        _ = reason


def test_kill_switch_aborts_state_machine(tmp_path: Path) -> None:
    run_list_file = tmp_path / "run_list.json"
    run_list_file.write_text(json.dumps({"run_list": []}), encoding="utf-8")

    controller = BotController(
        run_list_path=run_list_file,
        corruptos_csv_path=tmp_path / "corruptos.csv",
        ui_points={"cast_button": Point(1, 1)},
        input_driver=DryRunInputDriver(),
        dry_run=True,
    )
    ks = FakeKillSwitch()
    sm = FishingStateMachine(
        capture=DummyCapture(),
        tracker=DummyTracker(),
        classifier=DummyClassifier(),
        controller=controller,
        alert=DummyAlert(),
        kill_switch=ks,
        timeouts={"IDLE_OR_CAST": 5.0, "WAITING_MINIGAME": 20.0},
    )

    assert sm.step() == State.WAITING_MINIGAME
    ks.trigger()
    assert sm.step() == State.ABORTED
    assert sm.is_terminal is True


def test_setup_kill_switch_failure_warns_in_dry_run(caplog) -> None:
    ks = FakeKillSwitch(fail_on_register=True)
    with caplog.at_level("WARNING"):
        ok = setup_kill_switch(ks, dry_run=True)

    assert ok is True
    assert any("dry-run" in rec.message.lower() for rec in caplog.records)


def test_setup_kill_switch_failure_aborts_in_real_mode(caplog) -> None:
    ks = FakeKillSwitch(fail_on_register=True)
    with caplog.at_level("ERROR"):
        ok = setup_kill_switch(ks, dry_run=False)

    assert ok is False
    assert any("administrador" in rec.message.lower() for rec in caplog.records)
