"""Script auxiliar para extraer frames PNG sin pérdida desde video y recortar ROIs.

Permite preparar frames de referencia a resolución original (p. ej. 1680x976) y
generar recortes (crops) para plantillas en ``assets/templates/`` o inspección
visual con ``debug/visualizer.py``, sin ejecutar acciones de automatización.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from core.types import ROI
from debug.visualizer import Visualizer

logger = logging.getLogger(__name__)


def _format_timestamp_tag(seconds: float) -> str:
    """Formatea segundos como etiqueta segura para nombres de archivo (ej. 12.50 -> 0012_500s)."""
    whole = int(seconds)
    millis = round((seconds - whole) * 1000)
    if millis == 1000:
        whole += 1
        millis = 0
    return f"{whole:04d}_{millis:03d}s"


def extract_frames_by_timestamps(
    video_path: str | Path,
    output_dir: str | Path,
    timestamps: Sequence[float],
    prefix: str = "frame",
    expected_size: tuple[int, int] | None = None,
) -> list[Path]:
    """Extrae frames PNG sin pérdida en los timestamps (segundos) indicados.

    Lee secuencialmente por timestamp real (``CAP_PROP_POS_MSEC``) para ser preciso
    en grabaciones MP4 con tasa de cuadros variable (VFR) y conserva la resolución
    original del video sin reescalar.
    """
    src = Path(video_path)
    if not src.is_file():
        raise FileNotFoundError(f"No se encontró el video: {src}")

    targets: list[float] = []
    for raw_ts in timestamps:
        ts = float(raw_ts)
        if ts < 0:
            raise ValueError(f"Timestamp negativo no permitido: {ts}")
        targets.append(ts)
    if not targets:
        return []

    sorted_targets = sorted(set(targets))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video con OpenCV: {src}")

    saved_paths: list[Path] = []
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        logger.info("Video abierto: %s (%dx%d, %.2f fps nominal)", src.name, width, height, fps)
        if expected_size is not None and (width, height) != expected_size:
            logger.warning(
                "Resolución del video (%dx%d) distinta a la esperada (%dx%d).",
                width,
                height,
                expected_size[0],
                expected_size[1],
            )

        target_idx = 0
        frame_idx = 0
        last_sec = 0.0
        while target_idx < len(sorted_targets):
            if not cap.grab():
                break

            pos_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            if pos_msec > 0.0 or frame_idx == 0:
                current_sec = pos_msec / 1000.0
            elif fps > 0:
                current_sec = frame_idx / fps
            else:
                current_sec = 0.0
            last_sec = current_sec
            frame_idx += 1

            if current_sec + 1e-6 < sorted_targets[target_idx]:
                continue

            ok, frame = cap.retrieve()
            if not ok or frame is None:
                logger.warning("No se pudo decodificar el frame en t=%.3fs.", current_sec)
                continue

            while target_idx < len(sorted_targets) and current_sec + 1e-6 >= sorted_targets[target_idx]:
                requested_ts = sorted_targets[target_idx]
                out_path = out_dir / f"{prefix}_{_format_timestamp_tag(requested_ts)}.png"
                if not cv2.imwrite(str(out_path), frame):
                    raise RuntimeError(f"No se pudo escribir la imagen PNG: {out_path}")
                saved_paths.append(out_path)
                logger.info(
                    "Guardado %s (%dx%d, t_real=%.3fs)",
                    out_path,
                    frame.shape[1],
                    frame.shape[0],
                    current_sec,
                )
                target_idx += 1

        if target_idx < len(sorted_targets):
            omitted = [round(t, 3) for t in sorted_targets[target_idx:]]
            logger.warning(
                "Fin real del video en t=%.3fs (%d frames decodificados); "
                "se omiten timestamps posteriores: %s",
                last_sec,
                frame_idx,
                omitted,
            )
    finally:
        cap.release()

    return saved_paths


def extract_frames_by_interval(
    video_path: str | Path,
    output_dir: str | Path,
    interval_sec: float,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    prefix: str = "frame",
    expected_size: tuple[int, int] | None = None,
) -> list[Path]:
    """Extrae frames PNG sin pérdida cada ``interval_sec`` segundos entre ``start_sec`` y ``end_sec``."""
    if interval_sec <= 0:
        raise ValueError("interval_sec debe ser mayor a 0.")
    if start_sec < 0:
        raise ValueError("start_sec no puede ser negativo.")

    src = Path(video_path)
    if not src.is_file():
        raise FileNotFoundError(f"No se encontró el video: {src}")

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video con OpenCV: {src}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = (frame_count / fps) if fps > 0 and frame_count > 0 else 0.0
    finally:
        cap.release()

    limit = duration if end_sec is None else min(float(end_sec), duration) if duration > 0 else float(end_sec)
    if limit < start_sec:
        return []

    timestamps: list[float] = []
    current = float(start_sec)
    while current <= limit + 1e-9:
        timestamps.append(round(current, 6))
        current += interval_sec

    return extract_frames_by_timestamps(
        video_path=src,
        output_dir=output_dir,
        timestamps=timestamps,
        prefix=prefix,
        expected_size=expected_size,
    )


def crop_roi_from_image(
    image_path: str | Path,
    roi: ROI,
    output_path: str | Path,
    preview_path: str | Path | None = None,
    label: str = "ROI",
) -> Path:
    """Recorta una región ``ROI(x, y, width, height)`` de un PNG y la guarda sin pérdida.

    Si se indica ``preview_path``, guarda además una copia anotada del frame completo
    usando ``debug.visualizer.Visualizer`` para verificar visualmente el encuadre.
    """
    src = Path(image_path)
    if not src.is_file():
        raise FileNotFoundError(f"No se encontró la imagen: {src}")

    frame: np.ndarray | None = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"No se pudo leer la imagen con OpenCV: {src}")

    img_h, img_w = frame.shape[:2]
    x1, y1 = roi.x, roi.y
    x2, y2 = roi.x + roi.width, roi.y + roi.height
    if roi.width <= 0 or roi.height <= 0:
        raise ValueError(f"Ancho y alto del ROI deben ser > 0: {roi}")
    if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
        raise ValueError(
            f"ROI {roi} fuera de los límites de la imagen ({img_w}x{img_h})."
        )

    crop = frame[y1:y2, x1:x2]
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dest), crop):
        raise RuntimeError(f"No se pudo guardar el recorte PNG: {dest}")
    logger.info("Recorte guardado en %s (%dx%d)", dest, crop.shape[1], crop.shape[0])

    if preview_path is not None:
        preview_dest = Path(preview_path)
        preview_dest.parent.mkdir(parents=True, exist_ok=True)
        annotated = Visualizer().render_debug_frame(
            frame,
            rois={label: roi},
            labels=[f"{label}: x={roi.x}, y={roi.y}, w={roi.width}, h={roi.height}"],
        )
        if not cv2.imwrite(str(preview_dest), annotated):
            raise RuntimeError(f"No se pudo guardar el preview PNG: {preview_dest}")
        logger.info("Preview anotado guardado en %s", preview_dest)

    return dest


def build_arg_parser() -> argparse.ArgumentParser:
    """Construye la interfaz CLI con subcomandos ``extract`` y ``crop``."""
    parser = argparse.ArgumentParser(
        description="Extrae frames PNG sin pérdida desde video y recorta ROIs de referencia."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_p = subparsers.add_parser(
        "extract",
        help="Extrae frames PNG a resolución original por timestamps o intervalo.",
    )
    extract_p.add_argument("video", help="Ruta al archivo de video (.mp4).")
    extract_p.add_argument(
        "-o",
        "--output-dir",
        default="samples/frames",
        help="Carpeta de salida para los PNG (por defecto: samples/frames).",
    )
    mode_group = extract_p.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "-t",
        "--timestamps",
        type=float,
        nargs="+",
        help="Lista de timestamps en segundos (ej: --timestamps 1.5 4.2 10.0).",
    )
    mode_group.add_argument(
        "-i",
        "--interval",
        type=float,
        help="Intervalo en segundos entre frames extraídos (ej: --interval 2.0).",
    )
    extract_p.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="Segundo inicial para --interval (por defecto: 0.0).",
    )
    extract_p.add_argument(
        "--end",
        type=float,
        default=None,
        help="Segundo final opcional para --interval.",
    )
    extract_p.add_argument(
        "--prefix",
        default="frame",
        help="Prefijo para los archivos PNG generados (por defecto: frame).",
    )
    extract_p.add_argument(
        "--expected-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Resolución esperada para validar sin reescalar (ej: --expected-size 1680 976).",
    )

    crop_p = subparsers.add_parser(
        "crop",
        help="Recorta una región [x, y, w, h] desde un PNG extraído.",
    )
    crop_p.add_argument("image", help="Ruta al PNG de origen.")
    crop_p.add_argument(
        "--roi",
        type=int,
        nargs=4,
        required=True,
        metavar=("X", "Y", "W", "H"),
        help="Región a recortar en píxeles: X Y W H.",
    )
    crop_p.add_argument(
        "-o",
        "--output",
        required=True,
        help="Ruta del PNG recortado (ej: assets/templates/ui/fight.png).",
    )
    crop_p.add_argument(
        "--preview",
        default=None,
        help="Ruta opcional para guardar el frame completo con el ROI dibujado.",
    )
    crop_p.add_argument(
        "--label",
        default="ROI",
        help="Etiqueta para el preview anotado (por defecto: ROI).",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_arg_parser().parse_args(argv)

    if args.command == "extract":
        expected = tuple(args.expected_size) if args.expected_size else None
        if args.timestamps is not None:
            saved = extract_frames_by_timestamps(
                video_path=args.video,
                output_dir=args.output_dir,
                timestamps=args.timestamps,
                prefix=args.prefix,
                expected_size=expected,
            )
        else:
            saved = extract_frames_by_interval(
                video_path=args.video,
                output_dir=args.output_dir,
                interval_sec=args.interval,
                start_sec=args.start,
                end_sec=args.end,
                prefix=args.prefix,
                expected_size=expected,
            )
        logger.info("Total de frames guardados: %d", len(saved))
        return 0

    if args.command == "crop":
        x, y, w, h = args.roi
        crop_roi_from_image(
            image_path=args.image,
            roi=ROI(x=x, y=y, width=w, height=h),
            output_path=args.output,
            preview_path=args.preview,
            label=args.label,
        )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
