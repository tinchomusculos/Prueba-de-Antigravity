"""Herramientas de depuración visual sobre frames con OpenCV."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import cv2
import numpy as np

from core.types import Point, ROI

BGRColor = tuple[int, int, int]


class Visualizer:
    """Dibuja ROIs, centros, orden de puntos y etiquetas sobre frames BGR."""

    def __init__(self, font_scale: float = 0.55, thickness: int = 2) -> None:
        self.font_scale = font_scale
        self.thickness = thickness

    def draw_rois(
        self,
        frame: np.ndarray,
        rois: Mapping[str, ROI],
        color: BGRColor = (255, 200, 0),
    ) -> np.ndarray:
        """Dibuja rectángulos y nombres de ROIs sobre una copia del frame."""
        canvas = frame.copy()
        for name, roi in rois.items():
            top_left = (roi.x, roi.y)
            bottom_right = (roi.x + roi.width, roi.y + roi.height)
            cv2.rectangle(canvas, top_left, bottom_right, color, self.thickness)
            label_pos = (roi.x, max(16, roi.y - 6))
            cv2.putText(
                canvas,
                name,
                label_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                color,
                max(1, self.thickness - 1),
                cv2.LINE_AA,
            )
        return canvas

    def draw_ordered_points(
        self,
        frame: np.ndarray,
        points: Sequence[Point],
        start_color: BGRColor = (0, 255, 0),
        node_color: BGRColor = (0, 165, 255),
        line_color: BGRColor = (255, 255, 0),
    ) -> np.ndarray:
        """Dibuja los centros, el orden numérico y las líneas entre puntos del minijuego."""
        canvas = frame.copy()
        for idx, pt in enumerate(points):
            if idx > 0:
                prev = points[idx - 1]
                cv2.line(canvas, (prev.x, prev.y), (pt.x, pt.y), line_color, self.thickness)

        for idx, pt in enumerate(points):
            color = start_color if idx == 0 else node_color
            cv2.circle(canvas, (pt.x, pt.y), 7, color, -1)
            cv2.putText(
                canvas,
                str(idx + 1),
                (pt.x + 9, pt.y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                color,
                self.thickness,
                cv2.LINE_AA,
            )
        return canvas

    def draw_labels(
        self,
        frame: np.ndarray,
        labels: Sequence[str],
        origin: Point = Point(12, 26),
        line_height: int = 24,
        color: BGRColor = (255, 255, 255),
        bg_color: BGRColor | None = (20, 20, 20),
    ) -> np.ndarray:
        """Dibuja un bloque de etiquetas de estado/depuración sobre el frame."""
        canvas = frame.copy()
        for idx, text in enumerate(labels):
            y = origin.y + idx * line_height
            if bg_color is not None:
                (text_w, text_h), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1
                )
                cv2.rectangle(
                    canvas,
                    (origin.x - 4, y - text_h - 4),
                    (origin.x + text_w + 4, y + baseline + 4),
                    bg_color,
                    -1,
                )
            cv2.putText(
                canvas,
                text,
                (origin.x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                color,
                1,
                cv2.LINE_AA,
            )
        return canvas

    def render_debug_frame(
        self,
        frame: np.ndarray,
        rois: Mapping[str, ROI] | None = None,
        points: Sequence[Point] | None = None,
        labels: Sequence[str] | None = None,
    ) -> np.ndarray:
        """Compone ROIs, puntos ordenados y etiquetas en un único frame de depuración."""
        canvas = frame.copy()
        if rois:
            canvas = self.draw_rois(canvas, rois)
        if points:
            canvas = self.draw_ordered_points(canvas, points)
        if labels:
            canvas = self.draw_labels(canvas, labels)
        return canvas
