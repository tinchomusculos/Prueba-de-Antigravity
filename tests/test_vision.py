"""Pruebas sintéticas para los clasificadores de visión."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from core.types import Form, Point, ROI
from vision.encounter_classifier import EncounterClassifier
from vision.minigame_tracker import MinigameTracker
from vision.screen_capture import ScreenCapture


def _draw_minigame_frame() -> tuple[np.ndarray, ROI, list[Point]]:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:] = (18, 24, 28)
    roi = ROI(200, 150, 500, 420)

    cv2.rectangle(frame, (500, 65), (780, 94), (245, 245, 245), 2)
    cv2.rectangle(frame, (503, 68), (690, 91), (45, 220, 95), -1)

    local_nodes = {
        "green": (370, 200),
        "cursor_yellow": (280, 130),
        "white": (130, 140),
        "gray": (130, 320),
        "dark": (300, 330),
    }
    colors = {
        "green": (0, 255, 0),
        "cursor_yellow": (138, 255, 255),
        "white": (245, 245, 245),
        "gray": (100, 100, 100),
        "dark": (10, 10, 10),
    }
    for first, second, color in (
        ("green", "white", colors["green"]),
        ("white", "gray", colors["white"]),
        ("gray", "dark", (85, 85, 85)),
    ):
        start = tuple(
            local_nodes[first][axis] + (roi.x, roi.y)[axis] for axis in (0, 1)
        )
        end = tuple(
            local_nodes[second][axis] + (roi.x, roi.y)[axis] for axis in (0, 1)
        )
        cv2.line(frame, start, end, color, 8)

    points = {
        name: Point(x + roi.x, y + roi.y)
        for name, (x, y) in local_nodes.items()
    }
    for name, point in points.items():
        cv2.circle(frame, (point.x, point.y), 22, colors[name], -1)
    # El halo amarillo desconectado del cursor no debe formar parte del recorrido
    return frame, roi, [points["green"], points["white"], points["gray"], points["dark"]]


def test_minigame_nodes_are_ordered_by_start_next_and_connections() -> None:
    frame, roi, expected = _draw_minigame_frame()
    tracker = MinigameTracker(roi)

    assert tracker.is_minigame_active(frame)
    assert tracker.get_ordered_nodes(frame) == expected


def test_minigame_returns_no_nodes_when_bar_or_roi_is_invalid() -> None:
    frame, roi, _ = _draw_minigame_frame()
    tracker = MinigameTracker(roi)
    no_bar = frame.copy()
    no_bar[:120] = (18, 24, 28)
    invalid_roi_tracker = MinigameTracker(ROI(1200, 700, 200, 100))
    tiny_roi_tracker = MinigameTracker(ROI(10, 10, 1, 1))

    assert tracker.get_ordered_nodes(no_bar) == []
    assert tracker.is_minigame_active(np.zeros((0, 0, 3), dtype=np.uint8)) is False
    assert invalid_roi_tracker.get_ordered_nodes(frame) == []
    assert tiny_roi_tracker.get_ordered_nodes(frame) == []


def _pattern(seed: int, shape: tuple[int, int, int] = (24, 54, 3)) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)


def _write_template(directory: Path, name: str, template: np.ndarray) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(directory / f"{name}.png"), template)


def _classifier(
    templates_dir: Path, rois: dict[str, ROI] | None = None
) -> EncounterClassifier:
    return EncounterClassifier(
        templates_dir=templates_dir,
        window_size=(1280, 720),
        rois=rois
        or {
            "loomian_name": ROI(20, 20, 140, 70),
            "loomian_sprite": ROI(700, 100, 400, 400),
        },
        template_match_threshold=0.85,
        min_confidence=0.80,
    )


@pytest.mark.parametrize(("name", "seed"), [("Arketta", 1), ("Goppie", 2)])
def test_classifies_loomian_names_from_loaded_templates(
    tmp_path: Path, name: str, seed: int
) -> None:
    template = _pattern(seed)
    _write_template(tmp_path / "loomians", name, template)
    classifier = _classifier(tmp_path)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[25:49, 25:79] = template

    detected_name, confidence = classifier.classify_name(frame)

    assert detected_name == name
    assert confidence >= classifier.min_confidence


def test_classify_form_is_unknown_when_no_form_template_exists(tmp_path: Path) -> None:
    classifier = _classifier(tmp_path)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    form, confidence = classifier.classify_form(frame)

    assert form == Form.UNKNOWN
    assert confidence < classifier.min_confidence


def test_classify_form_returns_unknown_for_a_low_confidence_match(
    tmp_path: Path,
) -> None:
    _write_template(tmp_path / "forms", "normal", _pattern(4))
    classifier = _classifier(tmp_path)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    form, confidence = classifier.classify_form(frame)

    assert form == Form.UNKNOWN
    assert confidence < classifier.min_confidence


def test_ambiguous_name_templates_return_no_classification(tmp_path: Path) -> None:
    template = _pattern(7)
    _write_template(tmp_path / "loomians", "Arketta", template)
    _write_template(tmp_path / "loomians", "Goppie", template.copy())
    classifier = _classifier(tmp_path)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[25:49, 25:79] = template

    name, confidence = classifier.classify_name(frame)

    assert name is None
    assert confidence < classifier.min_confidence


def test_classify_encounter_rejects_mismatched_window_size(tmp_path: Path) -> None:
    classifier = _classifier(tmp_path)

    result = classifier.classify_encounter(np.zeros((600, 800, 3), dtype=np.uint8))

    assert result.form == Form.UNKNOWN
    assert result.name is None
    assert result.confidence == 0.0


def test_detects_battle_menu_and_fishing_return_templates(tmp_path: Path) -> None:
    ui_dir = tmp_path / "ui"
    templates = {
        "fight": _pattern(11),
        "items": _pattern(12),
        "run": _pattern(13),
        "loomians": _pattern(14),
        "fishing_return": _pattern(15),
    }
    for name, template in templates.items():
        _write_template(ui_dir, name, template)

    rois = {
        "battle_menu": ROI(200, 500, 880, 200),
        "fishing_return": ROI(20, 20, 160, 80),
    }
    classifier = _classifier(tmp_path, rois)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    battle_frame = np.zeros_like(frame)
    for idx, key in enumerate(("fight", "items", "run", "loomians")):
        template = templates[key]
        x = 210 + idx * 190
        battle_frame[510:510 + template.shape[0], x:x + template.shape[1]] = template
    assert classifier.detect_battle_menu(battle_frame)

    fishing_frame = np.zeros_like(frame)
    fishing_frame[22:46, 22:76] = templates["fishing_return"]
    assert classifier.detect_fishing_return(fishing_frame)


def test_detect_shard_drop_by_words_two_lines_mr_and_unknown(tmp_path: Path) -> None:
    shard_word = _pattern(20, shape=(24, 54, 3))
    attack_word = _pattern(21, shape=(24, 64, 3))
    speed_word = _pattern(22, shape=(24, 60, 3))
    m_glyph = _pattern(23, shape=(24, 22, 3))
    r_glyph = _pattern(24, shape=(24, 22, 3))

    _write_template(tmp_path / "ui", "shard_word", shard_word)
    _write_template(tmp_path / "shards", "attack", attack_word)
    _write_template(tmp_path / "shards", "speed", speed_word)
    _write_template(tmp_path / "shards", "m", m_glyph)
    _write_template(tmp_path / "shards", "r", r_glyph)

    classifier = _classifier(tmp_path, {"battle_banner": ROI(100, 560, 1000, 130)})

    # Caso 1: Attack M en la misma línea
    frame_am = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame_am[580:604, 700:764] = attack_word
    frame_am[580:604, 772:794] = m_glyph
    frame_am[580:604, 804:858] = shard_word
    assert classifier.detect_shard_drop(frame_am) == "Attack M Shard"

    # Caso 2: Attack R partido en 2 líneas ("Attack" al final de línea 1, "R Shard!" en línea 2)
    frame_ar_2line = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame_ar_2line[580:604, 1010:1074] = attack_word
    frame_ar_2line[630:654, 120:142] = r_glyph
    frame_ar_2line[630:654, 152:206] = shard_word
    assert classifier.detect_shard_drop(frame_ar_2line) == "Attack R Shard"

    # Caso 3: Speed Shard partido en 2 líneas ("Speed" en línea 1, "Shard" en línea 2)
    frame_speed = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame_speed[580:604, 980:1040] = speed_word
    frame_speed[630:654, 120:174] = shard_word
    assert classifier.detect_shard_drop(frame_speed) == "Speed Shard"

    # Caso 4: Drop confirmado ("Shard" visible) pero stat ilegible -> "Unknown"
    frame_unknown = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame_unknown[630:654, 150:204] = shard_word
    assert classifier.detect_shard_drop(frame_unknown) == "Unknown"

    # Caso 5: Sin banner de shard -> None
    assert classifier.detect_shard_drop(np.zeros((720, 1280, 3), dtype=np.uint8)) is None


def test_normal_palette_positive_evidence_rejects_recolor_and_missing(tmp_path: Path) -> None:
    sprite_roi = ROI(700, 100, 200, 200)
    frame_normal = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Bloque amarillo brillante simulando sprite normal
    frame_normal[140:210, 750:820] = (0, 230, 240)
    hist = EncounterClassifier.extract_sprite_palette(frame_normal[100:300, 700:900])
    assert hist is not None

    pal_dir = tmp_path / "forms" / "palettes"
    pal_dir.mkdir(parents=True, exist_ok=True)
    import json as _json

    (pal_dir / "goppie.json").write_text(
        _json.dumps({"histogram": hist.tolist()}),
        encoding="utf-8",
    )

    classifier = _classifier(
        tmp_path,
        {"loomian_name": ROI(20, 20, 140, 70), "loomian_sprite": sprite_roi},
    )

    form_ok, score_ok = classifier.classify_form(frame_normal, species_name="Goppie")
    assert form_ok == Form.NORMAL
    assert score_ok >= 0.99

    # Recoloreo del sprite -> debe caer bajo umbral y devolver Form.UNKNOWN (nunca falso NORMAL)
    frame_recolor = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame_recolor[140:210, 750:820] = (230, 200, 0)
    form_bad, _ = classifier.classify_form(frame_recolor, species_name="Goppie")
    assert form_bad == Form.UNKNOWN

    # Especie sin paleta de referencia -> Form.UNKNOWN
    form_missing, _ = classifier.classify_form(frame_normal, species_name="Arketta")
    assert form_missing == Form.UNKNOWN


def test_screen_capture_fallback_when_ensure_mss_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail_mss(self: ScreenCapture) -> object:
        raise RuntimeError("display no disponible")

    monkeypatch.setattr(ScreenCapture, "_ensure_mss", _fail_mss)

    dry_capture = ScreenCapture(
        idle_fps=10.0,
        solving_dots_fps=30.0,
        clock=lambda: 0.0,
        sleeper=lambda _: None,
        fallback_size=(1680, 976),
    )
    frame = dry_capture.grab()
    assert frame.shape == (976, 1680, 3)
    assert int(frame.sum()) == 0

    real_capture = ScreenCapture(
        idle_fps=10.0,
        solving_dots_fps=30.0,
        clock=lambda: 0.0,
        sleeper=lambda _: None,
        fallback_size=None,
    )
    with pytest.raises(RuntimeError, match="display no disponible"):
        real_capture.grab()

