"""Evalúa ``MinigameTracker`` sobre ``samples/frames*`` y compara con ``samples/ground_truth.json``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from core.types import ROI
from vision.minigame_tracker import MinigameTracker


@dataclass(frozen=True, slots=True)
class FrameEvaluation:
    """Resultado de evaluación de un frame individual."""

    clip: str
    frame_path: str
    detected_active: bool
    bar_ratio: float | None
    ordered_nodes_count: int


def discover_frame_dirs(samples_dir: Path) -> list[Path]:
    """Devuelve las carpetas ``samples/frames*`` ordenadas alfabéticamente."""
    if not samples_dir.is_dir():
        return []
    return sorted(
        p for p in samples_dir.iterdir() if p.is_dir() and p.name.startswith("frames")
    )


def load_minigame_roi(settings_path: Path) -> ROI:
    """Lee el ROI ``minigame`` desde ``config/settings.json``."""
    with settings_path.open("r", encoding="utf-8") as fh:
        settings = json.load(fh)
    cfg = settings["rois"]["minigame"]
    return ROI(
        x=int(cfg["x"]),
        y=int(cfg["y"]),
        width=int(cfg["width"]),
        height=int(cfg["height"]),
    )


def evaluate_frames(
    samples_dir: Path,
    tracker: MinigameTracker,
) -> list[FrameEvaluation]:
    """Ejecuta ``MinigameTracker.analyze`` sobre todos los PNG en ``samples/frames*``."""
    results: list[FrameEvaluation] = []
    for folder in discover_frame_dirs(samples_dir):
        for img_path in sorted(folder.glob("*.png")):
            frame = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            rel_key = img_path.relative_to(samples_dir).as_posix()
            if frame is None:
                results.append(
                    FrameEvaluation(
                        clip=folder.name,
                        frame_path=rel_key,
                        detected_active=False,
                        bar_ratio=None,
                        ordered_nodes_count=0,
                    )
                )
                continue

            state = tracker.analyze(frame)
            results.append(
                FrameEvaluation(
                    clip=folder.name,
                    frame_path=rel_key,
                    detected_active=state.active,
                    bar_ratio=state.bar_ratio,
                    ordered_nodes_count=len(state.ordered_nodes),
                )
            )
    return results


def build_ground_truth_template(evaluations: Sequence[FrameEvaluation]) -> dict[str, Any]:
    """Genera una plantilla JSON de verdad terreno para que el usuario la complete."""
    frames: dict[str, dict[str, Any]] = {}
    for ev in evaluations:
        frames[ev.frame_path] = {
            "active": None,
            "expected_nodes": None,
        }
    return {
        "_instructions": (
            "Completa 'active' (true/false) y 'expected_nodes' (entero o null si no aplica) "
            "para evaluar precisión, recall y exactitud de conteo de nodos."
        ),
        "frames": frames,
    }


def _compare_with_ground_truth(
    evaluations: Sequence[FrameEvaluation],
    gt_path: Path,
) -> list[str]:
    lines: list[str] = []
    if not gt_path.is_file():
        lines.append(
            f"Ground truth: {gt_path} no encontrado (usa --init-ground-truth para crear plantilla)."
        )
        return lines

    with gt_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    gt_frames = raw.get("frames", {}) if isinstance(raw, dict) else {}

    labeled_active = 0
    tp = fp = fn = tn = 0
    labeled_nodes = 0
    exact_nodes = 0
    mismatches: list[str] = []

    for ev in evaluations:
        entry = gt_frames.get(ev.frame_path)
        if not isinstance(entry, dict):
            continue
        gt_active = entry.get("active")
        gt_nodes = entry.get("expected_nodes")

        if isinstance(gt_active, bool):
            labeled_active += 1
            if ev.detected_active and gt_active:
                tp += 1
            elif ev.detected_active and not gt_active:
                fp += 1
                mismatches.append(f"  FP active: {ev.frame_path} (pred=True, gt=False)")
            elif not ev.detected_active and gt_active:
                fn += 1
                mismatches.append(f"  FN active: {ev.frame_path} (pred=False, gt=True)")
            else:
                tn += 1

        if isinstance(gt_nodes, int):
            labeled_nodes += 1
            if ev.ordered_nodes_count == gt_nodes:
                exact_nodes += 1
            else:
                mismatches.append(
                    f"  Nodos != : {ev.frame_path} (pred={ev.ordered_nodes_count}, gt={gt_nodes})"
                )

    lines.append(f"Comparación contra {gt_path}:")
    if labeled_active == 0 and labeled_nodes == 0:
        lines.append(
            "  0 frames etiquetados todavía (todos los campos 'active' / 'expected_nodes' están en null)."
        )
        return lines

    if labeled_active > 0:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        accuracy = (tp + tn) / labeled_active
        lines.append(
            f"  Detección active ({labeled_active} etiquetados): "
            f"TP={tp} FP={fp} FN={fn} TN={tn} | "
            f"Precision={precision:.1%} Recall={recall:.1%} Accuracy={accuracy:.1%}"
        )
    if labeled_nodes > 0:
        node_acc = exact_nodes / labeled_nodes
        lines.append(
            f"  Conteo exacto de nodos ({labeled_nodes} etiquetados): "
            f"{exact_nodes}/{labeled_nodes} ({node_acc:.1%})"
        )
    if mismatches:
        lines.append("  Discrepancias:")
        lines.extend(mismatches)
    return lines


def format_report(
    evaluations: Sequence[FrameEvaluation],
    gt_path: Path | None = None,
) -> str:
    """Formatea un resumen por clip, detalle de frames activos y métricas vs. verdad terreno."""
    by_clip: dict[str, list[FrameEvaluation]] = {}
    for ev in evaluations:
        by_clip.setdefault(ev.clip, []).append(ev)

    lines: list[str] = ["=== Reporte de MinigameTracker sobre samples/frames* ==="]
    total_frames = len(evaluations)
    total_active = sum(1 for ev in evaluations if ev.detected_active)
    lines.append(f"Total analizado: {total_frames} frames | Activos detectados: {total_active}")
    lines.append("")

    for clip, items in by_clip.items():
        active_items = [ev for ev in items if ev.detected_active]
        total_nodes = sum(ev.ordered_nodes_count for ev in active_items)
        lines.append(
            f"[{clip}] frames={len(items)} | activos={len(active_items)} | nodos_ordenados_totales={total_nodes}"
        )
        for ev in active_items:
            bar_str = f"{ev.bar_ratio:.3f}" if ev.bar_ratio is not None else "None"
            lines.append(
                f"  - {ev.frame_path}: bar_ratio={bar_str}, ordered_nodes={ev.ordered_nodes_count}"
            )

    if gt_path is not None:
        lines.append("")
        lines.extend(_compare_with_ground_truth(evaluations, gt_path))

    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    """Construye el parser CLI de evaluación del tracker."""
    parser = argparse.ArgumentParser(
        description="Evalúa MinigameTracker sobre samples/frames* y compara con ground_truth.json."
    )
    parser.add_argument(
        "--samples-dir",
        default="samples",
        help="Directorio raíz con subcarpetas frames* (por defecto: samples).",
    )
    parser.add_argument(
        "--settings",
        default="config/settings.json",
        help="Ruta a config/settings.json para leer rois.minigame.",
    )
    parser.add_argument(
        "--ground-truth",
        default="samples/ground_truth.json",
        help="Ruta al JSON de verdad terreno (por defecto: samples/ground_truth.json).",
    )
    parser.add_argument(
        "--init-ground-truth",
        action="store_true",
        help="Genera (si no existe o con --overwrite) la plantilla samples/ground_truth.json.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobrescribe el archivo de verdad terreno al usar --init-ground-truth.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada CLI."""
    args = build_arg_parser().parse_args(argv)
    samples_dir = Path(args.samples_dir)
    settings_path = Path(args.settings)
    gt_path = Path(args.ground_truth)

    roi = load_minigame_roi(settings_path)
    tracker = MinigameTracker(roi)
    evaluations = evaluate_frames(samples_dir, tracker)

    if args.init_ground_truth:
        if gt_path.exists() and not args.overwrite:
            print(f"{gt_path} ya existe; usa --overwrite para regenerar.", file=sys.stderr)
        else:
            gt_path.parent.mkdir(parents=True, exist_ok=True)
            template = build_ground_truth_template(evaluations)
            gt_path.write_text(json.dumps(template, indent=2) + "\n", encoding="utf-8")
            print(f"Plantilla generada en {gt_path} ({len(evaluations)} frames).")

    print(format_report(evaluations, gt_path=gt_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
