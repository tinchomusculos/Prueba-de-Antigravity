"""Tests para core/controller.py: hot reload de run_list, escritura atómica y CSV de corruptos."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from core.controller import BotController, DryRunInputDriver
from core.types import Point

UI_POINTS = {
    "cast_button": Point(100, 100),
    "fight_button": Point(200, 200),
    "run_button": Point(300, 300),
    "corrupt_move_slot": Point(250, 250),
    "item_notice_close": Point(400, 400),
}


def test_run_list_hot_reload(tmp_path: Path) -> None:
    run_list_file = tmp_path / "run_list.json"
    run_list_file.write_text(json.dumps({"run_list": ["Goppie"]}), encoding="utf-8")

    controller = BotController(
        run_list_path=run_list_file,
        corruptos_csv_path=tmp_path / "corruptos.csv",
        ui_points=UI_POINTS,
        dry_run=True,
    )

    assert controller.is_in_run_list("Goppie") is True
    assert controller.is_in_run_list("Cavenish") is False

    # Edición externa en caliente sin reiniciar el controlador
    run_list_file.write_text(
        json.dumps({"run_list": ["Goppie", "Cavenish"]}, indent=2),
        encoding="utf-8",
    )

    assert controller.is_in_run_list("cavenish") is True


def test_add_to_run_list_case_insensitive_and_formatted(tmp_path: Path) -> None:
    run_list_file = tmp_path / "run_list.json"
    run_list_file.write_text(
        json.dumps({"run_list": ["Goppie", "Cavenish"]}, indent=2) + "\n",
        encoding="utf-8",
    )

    controller = BotController(
        run_list_path=run_list_file,
        corruptos_csv_path=tmp_path / "corruptos.csv",
        ui_points=UI_POINTS,
        dry_run=True,
    )

    assert controller.add_to_run_list("goppie") is False
    assert controller.add_to_run_list("   ") is False
    assert controller.add_to_run_list("Kyeggo") is True
    assert controller.add_to_run_list("KYEGGO") is False

    raw = run_list_file.read_text(encoding="utf-8")
    assert '  "run_list": [' in raw
    assert raw.endswith("\n")
    assert json.loads(raw) == {"run_list": ["Goppie", "Cavenish", "Kyeggo"]}


def test_append_corrupt_creates_header_and_appends_iso8601(tmp_path: Path) -> None:
    csv_file = tmp_path / "data" / "corruptos.csv"
    fixed_dt = datetime(2026, 11, 22, 13, 45, 0, tzinfo=timezone.utc)

    controller = BotController(
        run_list_path=tmp_path / "run_list.json",
        corruptos_csv_path=csv_file,
        ui_points=UI_POINTS,
        dry_run=True,
        now_provider=lambda: fixed_dt,
    )

    controller.append_corrupt("Goppie", "Health Shard")
    controller.append_corrupt("Cavenish", "Speed Shard")

    with csv_file.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))

    assert rows == [
        ["timestamp", "loomian", "shard"],
        ["2026-11-22T13:45:00+00:00", "Goppie", "Health Shard"],
        ["2026-11-22T13:45:00+00:00", "Cavenish", "Speed Shard"],
    ]


def test_dry_run_driver_records_actions(tmp_path: Path) -> None:
    driver = DryRunInputDriver()
    controller = BotController(
        run_list_path=tmp_path / "run_list.json",
        corruptos_csv_path=tmp_path / "corruptos.csv",
        ui_points=UI_POINTS,
        input_driver=driver,
        dry_run=True,
    )

    controller.cast_rod()
    controller.click_run()
    controller.move_to_dot(Point(50, 60))

    assert len(driver.recorded_actions) == 3
    assert driver.recorded_actions[0][0] == "click"
    assert driver.recorded_actions[2] == ("move_to", Point(50, 60))


def test_window_origin_offset_applied_to_driver(tmp_path: Path) -> None:
    driver = DryRunInputDriver()
    controller = BotController(
        run_list_path=tmp_path / "run_list.json",
        corruptos_csv_path=tmp_path / "corruptos.csv",
        ui_points=UI_POINTS,
        input_driver=driver,
        dry_run=True,
        window_origin=Point(0, 32),
    )

    controller.click_run()
    controller.move_to_dot(Point(50, 60))

    assert driver.recorded_actions == [
        ("click", (Point(300, 332), "left")),
        ("move_to", Point(50, 92)),
    ]

