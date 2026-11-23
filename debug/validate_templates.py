"""Valida cada template de ``assets/templates/`` sobre TODOS los frames de ``samples/frames*``."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2

from core.types import ROI
from vision.encounter_classifier import EncounterClassifier


@dataclass(frozen=True, slots=True)
class TemplateValidationRow:
    """Resultado de validación de un template frente al conjunto completo de frames."""

    category: str
    name: str
    roi_name: str
    positives: int
    false_positives: int
    false_negatives: int
    max_fp_score: float
    min_tp_score: float | None
    matched_frames: tuple[str, ...]
    fp_frames: tuple[str, ...]
    fn_frames: tuple[str, ...]


# Etiquetas observadas en los 55 frames actuales de samples/frames* para medir TP/FP/FN reales.
# Se usan sets explícitos de los frames donde cada UI/especie aparece en el menú o banner.
EXPECTED_FRAMES: dict[str, set[str]] = {
    "ui/fight": {
        "frames/frame_0004_000s.png",
        "frames/frame_0005_000s.png",
        "frames/frame_0005_500s.png",
        "frames/frame_0006_000s.png",
        "frames/frame_0007_000s.png",
        "frames/frame_0008_000s.png",
        "frames/frame_0009_000s.png",
        "frames_clip2/clip2_0005_000s.png",
        "frames_clip3/clip3_0007_500s.png",
        "frames_clip3/clip3_0008_000s.png",
        "frames_clip3/clip3_0008_500s.png",
        "frames_clip3/clip3_0009_000s.png",
        "frames_clip3/clip3_0009_500s.png",
        "frames_clip3/clip3_0010_000s.png",
        "frames_clip3/clip3_0010_500s.png",
        "frames_clip3/clip3_0011_000s.png",
        "frames_clip3/clip3_0011_500s.png",
        "frames_clip3/clip3_0012_000s.png",
        "frames_clip3/clip3_0012_500s.png",
        "frames_clip3/clip3_0013_000s.png",
    },
    "ui/items": {
        "frames/frame_0004_000s.png",
        "frames/frame_0009_000s.png",
        "frames_clip2/clip2_0005_000s.png",
        "frames_clip3/clip3_0007_500s.png",
        "frames_clip3/clip3_0008_000s.png",
        "frames_clip3/clip3_0008_500s.png",
        "frames_clip3/clip3_0013_000s.png",
    },
    "ui/run": {
        "frames/frame_0004_000s.png",
        "frames/frame_0009_000s.png",
        "frames_clip2/clip2_0005_000s.png",
        "frames_clip3/clip3_0007_500s.png",
        "frames_clip3/clip3_0008_000s.png",
        "frames_clip3/clip3_0008_500s.png",
        "frames_clip3/clip3_0013_000s.png",
    },
    "ui/loomians": {
        "frames/frame_0004_000s.png",
        "frames/frame_0009_000s.png",
        "frames_clip2/clip2_0005_000s.png",
        "frames_clip3/clip3_0007_500s.png",
        "frames_clip3/clip3_0008_000s.png",
        "frames_clip3/clip3_0008_500s.png",
        "frames_clip3/clip3_0013_000s.png",
    },
    "ui/cancel": {
        "frames/frame_0005_500s.png",
        "frames/frame_0006_000s.png",
        "frames/frame_0007_000s.png",
        "frames/frame_0008_000s.png",
        "frames_clip3/clip3_0009_500s.png",
        "frames_clip3/clip3_0010_000s.png",
        "frames_clip3/clip3_0010_500s.png",
        "frames_clip3/clip3_0011_000s.png",
        "frames_clip3/clip3_0011_500s.png",
        "frames_clip3/clip3_0012_000s.png",
        "frames_clip3/clip3_0012_500s.png",
    },
    "ui/got_away_safely": {
        "frames/frame_0010_000s.png",
        "frames/frame_0011_000s.png",
        "frames_clip2/clip2_0006_000s.png",
        "frames_clip3/clip3_0014_000s.png",
        "frames_clip3/clip3_0014_500s.png",
        "frames_clip3/clip3_0015_000s.png",
    },
    "ui/fishing_return": {
        "frames_clip2/clip2_0000_000s.png",
        "frames_clip3/clip3_0000_000s.png",
        "frames_clip3/clip3_0000_500s.png",
        "frames_clip3/clip3_0001_000s.png",
        "frames_clip3/clip3_0001_500s.png",
        "frames_clip3/clip3_0002_000s.png",
    },
    "loomians/goppie": {
        "frames/frame_0004_000s.png",
        "frames/frame_0005_000s.png",
        "frames/frame_0005_500s.png",
        "frames/frame_0006_000s.png",
        "frames/frame_0007_000s.png",
        "frames/frame_0008_000s.png",
        "frames/frame_0009_000s.png",
        "frames/frame_0010_000s.png",
        "frames/frame_0011_000s.png",
        "frames/frame_0011_500s.png",
        "frames_clip3/clip3_0007_500s.png",
        "frames_clip3/clip3_0008_000s.png",
        "frames_clip3/clip3_0008_500s.png",
        "frames_clip3/clip3_0009_000s.png",
        "frames_clip3/clip3_0009_500s.png",
        "frames_clip3/clip3_0010_000s.png",
        "frames_clip3/clip3_0010_500s.png",
        "frames_clip3/clip3_0011_000s.png",
        "frames_clip3/clip3_0011_500s.png",
        "frames_clip3/clip3_0012_000s.png",
        "frames_clip3/clip3_0012_500s.png",
        "frames_clip3/clip3_0013_000s.png",
        "frames_clip3/clip3_0013_500s.png",
        "frames_clip3/clip3_0014_000s.png",
        "frames_clip3/clip3_0014_500s.png",
        "frames_clip3/clip3_0015_000s.png",
        "frames_clip3/clip3_0015_500s.png",
    },
    "loomians/arketta": {
        "frames_clip2/clip2_0005_000s.png",
        "frames_clip2/clip2_0006_000s.png",
        "frames_clip2/clip2_0007_000s.png",
    },
}


def _discover_frames(samples_dir: Path) -> list[tuple[str, cv2.typing.MatLike]]:
    loaded: list[tuple[str, cv2.typing.MatLike]] = []
    for folder in sorted(samples_dir.iterdir()):
        if not folder.is_dir() or not folder.name.startswith("frames"):
            continue
        for img_path in sorted(folder.glob("*.png")):
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is not None:
                loaded.append((img_path.relative_to(samples_dir).as_posix(), img))
    return loaded


def _roi_for_template(category: str, name: str) -> str:
    if category == "loomians":
        return "loomian_name"
    if category == "forms":
        return "loomian_sprite"
    if category == "shards":
        return "battle_banner"
    if name in {"fight", "items", "run", "loomians", "cancel"}:
        return "battle_menu"
    if name in {"corrupt_powerful", "corrupt_appeared", "shard_word", "dropped_a", "shard_drop"}:
        return "battle_banner"
    return "__full__"


def validate_all_templates(
    settings_path: Path,
    samples_dir: Path,
) -> tuple[list[TemplateValidationRow], int]:
    """Valida todos los templates de ``templates_dir`` contra ``samples/frames*``."""
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    rois = {k: ROI(**v) for k, v in settings["rois"].items()}
    window_size = (
        int(settings["window_size"]["width"]),
        int(settings["window_size"]["height"]),
    )
    classifier = EncounterClassifier(
        templates_dir=settings["templates_dir"],
        window_size=window_size,
        rois=rois,
        template_match_threshold=float(settings["template_match_threshold"]),
        min_confidence=float(settings["min_confidence"]),
    )
    threshold = max(classifier.template_match_threshold, classifier.min_confidence)
    frames = _discover_frames(samples_dir)

    buckets: list[tuple[str, dict[str, cv2.typing.MatLike]]] = [
        ("ui", classifier._ui_templates),
        ("loomians", classifier._loomian_templates),
        ("forms", classifier._form_templates),
        ("shards", classifier._shard_templates),
    ]

    rows: list[TemplateValidationRow] = []
    for category, tpl_map in buckets:
        for name, template in sorted(tpl_map.items()):
            roi_name = _roi_for_template(category, name)
            expected = EXPECTED_FRAMES.get(f"{category}/{name}", set())
            matched: list[str] = []
            fp_list: list[str] = []
            fn_list: list[str] = []
            max_fp_score = 0.0
            min_tp_score: float | None = None

            for rel_path, frame in frames:
                if roi_name == "__full__":
                    region = frame[:, :, :3]
                else:
                    region = classifier._crop_roi(frame, roi_name)
                score = classifier._match_template(region, template) if region is not None else 0.0
                hit = score >= threshold
                if hit:
                    matched.append(rel_path)

                if rel_path in expected:
                    if hit:
                        min_tp_score = score if min_tp_score is None else min(min_tp_score, score)
                    else:
                        fn_list.append(f"{rel_path} ({score:.3f})")
                else:
                    if hit:
                        fp_list.append(f"{rel_path} ({score:.3f})")
                    max_fp_score = max(max_fp_score, score)

            rows.append(
                TemplateValidationRow(
                    category=category,
                    name=name,
                    roi_name=roi_name,
                    positives=len(matched) - len(fp_list),
                    false_positives=len(fp_list),
                    false_negatives=len(fn_list),
                    max_fp_score=max_fp_score,
                    min_tp_score=min_tp_score,
                    matched_frames=tuple(matched),
                    fp_frames=tuple(fp_list),
                    fn_frames=tuple(fn_list),
                )
            )
    return rows, len(frames)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Valida todos los templates de assets/templates contra samples/frames*."
    )
    parser.add_argument("--settings", default="config/settings.json")
    parser.add_argument("--samples-dir", default="samples")
    args = parser.parse_args(argv)

    rows, frame_count = validate_all_templates(Path(args.settings), Path(args.samples_dir))
    print(f"=== Validación de {len(rows)} templates sobre {frame_count} frames ===")
    any_fp = False
    for row in rows:
        tp_str = f"{row.min_tp_score:.3f}" if row.min_tp_score is not None else "N/A"
        print(
            f"- {row.category}/{row.name:16s} [roi={row.roi_name:12s}] | "
            f"TP={row.positives:2d} FP={row.false_positives} FN={row.false_negatives} | "
            f"min_TP={tp_str} max_FP={row.max_fp_score:.3f}"
        )
        if row.fp_frames:
            any_fp = True
            print(f"    Falsos positivos: {', '.join(row.fp_frames)}")
        if row.fn_frames:
            print(f"    Falsos negativos: {', '.join(row.fn_frames)}")

    if any_fp:
        print("ERROR: Se detectaron falsos positivos en al menos un template.")
        return 1
    print("OK: 0 falsos positivos en todos los templates validados.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
