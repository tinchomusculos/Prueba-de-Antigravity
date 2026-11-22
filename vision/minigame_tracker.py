"""Seguimiento visual de la barra y los puntos del minijuego de pesca."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.types import Point, ROI


@dataclass(frozen=True, slots=True)
class MinigameState:
    """Estado instantáneo detectado del minijuego de puntos."""

    active: bool
    bar_ratio: float | None = None
    start_node: Point | None = None
    ordered_nodes: tuple[Point, ...] = ()


@dataclass(frozen=True, slots=True)
class _Node:
    point: Point
    color: str
    radius: float
    brightness: float


class MinigameTracker:
    """Detecta la barra, los nodos y el recorrido conectado usando color, contornos y líneas."""

    def __init__(self, minigame_roi: ROI) -> None:
        self.minigame_roi = minigame_roi
        self._cached_frame: np.ndarray | None = None
        self._cached_bar: tuple[float, tuple[int, int, int, int]] | None = None
        self._bar_ready = False
        self._cached_nodes: tuple[_Node, ...] = ()
        self._nodes_ready = False

    def _prepare_cache(self, frame: np.ndarray) -> None:
        if frame is self._cached_frame:
            return
        self._cached_frame = frame
        self._cached_bar = None
        self._bar_ready = False
        self._cached_nodes = ()
        self._nodes_ready = False

    @staticmethod
    def _valid_frame(frame: np.ndarray) -> bool:
        return (
            isinstance(frame, np.ndarray)
            and frame.dtype == np.uint8
            and frame.ndim == 3
            and frame.shape[2] >= 3
            and frame.shape[0] > 0
            and frame.shape[1] > 0
        )

    def _crop_roi(self, frame: np.ndarray) -> np.ndarray | None:
        if not self._valid_frame(frame):
            return None

        roi = self.minigame_roi
        if roi.x < 0 or roi.y < 0 or roi.width <= 0 or roi.height <= 0:
            return None
        if roi.x + roi.width > frame.shape[1] or roi.y + roi.height > frame.shape[0]:
            return None

        crop = frame[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width, :3]
        return crop if crop.size else None

    @staticmethod
    def _green_bar_mask(image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, np.array((40, 100, 120)), np.array((88, 255, 255)))

    @staticmethod
    def _green_node_mask(hsv: np.ndarray) -> np.ndarray:
        """Máscara estricta para el nodo verde brillante (excluye agua verde-azulada y pasto oscuro)."""
        return cv2.inRange(hsv, np.array((45, 120, 165)), np.array((82, 255, 255)))

    def _find_bar(self, frame: np.ndarray) -> tuple[float, tuple[int, int, int, int]] | None:
        if not self._valid_frame(frame):
            return None
        self._prepare_cache(frame)
        if self._bar_ready:
            return self._cached_bar

        height, width = frame.shape[:2]
        top_height = max(1, int(height * 0.30))
        top = frame[:top_height, :, :3]
        outline = self._find_bar_outline(top)
        if outline is not None:
            x, y, box_width, box_height = outline
            green = self._green_bar_mask(top)
            inset_x = max(1, int(box_height * 0.08))
            inset_y = max(1, int(box_height * 0.08))
            inner = green[
                y + inset_y : y + box_height - inset_y,
                x + inset_x : x + box_width - inset_x,
            ]
            filled_columns = np.flatnonzero(np.any(inner > 0, axis=0)) if inner.size else []
            if len(filled_columns) >= max(12, int(box_width * 0.04)):
                filled_width = int(filled_columns[-1] - filled_columns[0] + 1)
                ratio = float(np.clip(filled_width / max(1, box_width - 2 * inset_x), 0.0, 1.0))
                self._cached_bar = (ratio, (x, y, box_width, box_height))
                self._bar_ready = True
                return self._cached_bar

        mask = self._green_bar_mask(top)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3)),
        )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_width = max(25, int(width * 0.05))
        candidates: list[tuple[int, int, int, int]] = []
        for contour in contours:
            x, y, box_width, box_height = cv2.boundingRect(contour)
            if (
                box_width < min_width
                or box_height < 4
                or box_height > max(70, int(height * 0.08))
                or box_width / max(1, box_height) < 4.0
                or y > top_height * 0.75
                or box_width > width * 0.9
            ):
                continue
            candidates.append((x, y, box_width, box_height))

        if not candidates:
            self._bar_ready = True
            return None
        x, y, filled_width, box_height = max(candidates, key=lambda box: box[2] * box[3])
        total_width = max(filled_width, int(width * 0.22))
        ratio = float(np.clip(filled_width / total_width, 0.0, 1.0))
        self._cached_bar = (ratio, (x, y, filled_width, box_height))
        self._bar_ready = True
        return self._cached_bar

    @staticmethod
    def _find_bar_outline(top: np.ndarray) -> tuple[int, int, int, int] | None:
        hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(
            hsv, np.array((0, 0, 190)), np.array((180, 90, 255))
        )
        kernel_width = max(20, int(top.shape[1] * 0.08))
        horizontal = cv2.morphologyEx(
            white_mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1)),
        )
        contours, _ = cv2.findContours(
            horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        segments: list[tuple[int, int, int]] = []
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            if width >= max(25, top.shape[1] * 0.08) and height <= 8:
                segments.append((x, y, width))

        rows: list[tuple[int, int, int]] = []
        for x, y, width in sorted(segments, key=lambda segment: segment[1]):
            matched_row = next(
                (idx for idx, row in enumerate(rows) if abs(row[1] - y) <= 3),
                None,
            )
            if matched_row is None:
                rows.append((x, y, width))
                continue
            old_x, old_y, old_width = rows[matched_row]
            left = min(old_x, x)
            right = max(old_x + old_width, x + width)
            rows[matched_row] = (left, min(old_y, y), right - left)

        best: tuple[int, int, int, int] | None = None
        for idx, (x1, y1, width1) in enumerate(rows):
            for x2, y2, width2 in rows[idx + 1 :]:
                distance_y = abs(y1 - y2)
                if not 8 <= distance_y <= 90 or min(width1, width2) < top.shape[1] * 0.12:
                    continue
                overlap = min(x1 + width1, x2 + width2) - max(x1, x2)
                if overlap < min(width1, width2) * 0.70:
                    continue
                left = min(x1, x2)
                right = max(x1 + width1, x2 + width2)
                candidate = (left, min(y1, y2), right - left, distance_y + 1)
                if best is None or candidate[2] > best[2]:
                    best = candidate
        return best

    def detect_time_bar(self, frame: np.ndarray) -> float | None:
        """Estima la proporción verde restante de la barra horizontal superior."""
        found = self._find_bar(frame)
        return None if found is None else found[0]

    @staticmethod
    def _circle_stats(
        crop: np.ndarray,
        mask: np.ndarray,
        center_x: float,
        center_y: float,
        radius: float,
    ) -> tuple[float, float, float]:
        """Devuelve (fill_ratio, interior_std_per_channel, mean_brightness) del disco interior."""
        inner_radius = max(2, int(round(radius * 0.60)))
        x1 = max(0, int(center_x) - inner_radius)
        x2 = min(mask.shape[1], int(center_x) + inner_radius + 1)
        y1 = max(0, int(center_y) - inner_radius)
        y2 = min(mask.shape[0], int(center_y) + inner_radius + 1)
        if x1 >= x2 or y1 >= y2:
            return 0.0, 255.0, 0.0
        yy, xx = np.ogrid[y1:y2, x1:x2]
        disk = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= inner_radius**2
        mask_vals = mask[y1:y2, x1:x2][disk]
        fill_ratio = float(np.count_nonzero(mask_vals)) / max(1, mask_vals.size)
        pixels = crop[y1:y2, x1:x2][disk].astype(np.float32)
        if pixels.size == 0:
            return fill_ratio, 255.0, 0.0
        channel_std = float(np.mean(np.std(pixels, axis=0)))
        mean_brightness = float(np.mean(pixels))
        return fill_ratio, channel_std, mean_brightness

    def _detect_nodes(self, frame: np.ndarray) -> list[_Node]:
        if not self._valid_frame(frame):
            return []
        self._prepare_cache(frame)
        if self._nodes_ready:
            return list(self._cached_nodes)

        crop = self._crop_roi(frame)
        if crop is None or min(crop.shape[:2]) < 24:
            self._cached_nodes = ()
            self._nodes_ready = True
            return []

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Bandas separadas: verde brillante inicial + 4 niveles acromáticos (blanco, gris claro, gris medio, negro)
        band_masks: list[tuple[str, np.ndarray]] = [
            ("green", self._green_node_mask(hsv)),
            ("neutral", cv2.inRange(hsv, np.array((0, 0, 225)), np.array((180, 28, 255)))),
            ("neutral", cv2.inRange(hsv, np.array((0, 0, 145)), np.array((180, 28, 224)))),
            ("neutral", cv2.inRange(hsv, np.array((0, 0, 55)), np.array((180, 28, 144)))),
            ("neutral", cv2.inRange(hsv, np.array((0, 0, 0)), np.array((180, 28, 54)))),
        ]

        min_dimension = min(crop.shape[:2])
        open_size = max(5, min(25, int(round(min_dimension * 0.045))))
        if open_size % 2 == 0:
            open_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
        min_radius = max(10.0, min_dimension * 0.032)
        max_radius = min_dimension * 0.14
        min_area = np.pi * (min_radius * 0.85) ** 2
        max_area = np.pi * (max_radius * 1.15) ** 2
        nodes: list[_Node] = []

        for color, mask in band_masks:
            opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(
                opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                area = cv2.contourArea(contour)
                if not min_area <= area <= max_area:
                    continue
                _, _, width, height = cv2.boundingRect(contour)
                if min(width, height) <= 0:
                    continue
                aspect_ratio = width / height
                if not 0.75 <= aspect_ratio <= 1.33:
                    continue
                (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
                if not min_radius <= radius <= max_radius:
                    continue
                circularity = area / max(1.0, np.pi * radius * radius)
                if circularity < 0.56:
                    continue
                fill_ratio, interior_std, mean_brightness = self._circle_stats(
                    crop, mask, center_x, center_y, radius
                )
                if fill_ratio < (0.85 if color == "green" else 0.78):
                    continue
                # Los nodos reales son círculos planos uniformes; rechaza texturas de agua/suelo
                if interior_std > (18.0 if color == "green" else 14.0):
                    continue

                center = Point(int(round(center_x)), int(round(center_y)))
                if any(
                    np.hypot(center.x - node.point.x, center.y - node.point.y)
                    < max(10.0, min(radius, node.radius) * 0.70)
                    for node in nodes
                ):
                    continue
                nodes.append(_Node(center, color, float(radius), mean_brightness))

        filtered_nodes: list[_Node] = []
        green_nodes = [node for node in nodes if node.color == "green"]
        if green_nodes:
            filtered_nodes.append(max(green_nodes, key=lambda node: node.radius))
        filtered_nodes.extend(node for node in nodes if node.color == "neutral")
        self._cached_nodes = tuple(filtered_nodes)
        self._nodes_ready = True
        return list(self._cached_nodes)

    def get_detected_nodes(self, frame: np.ndarray) -> list[Point]:
        """Devuelve los centros detectados en coordenadas absolutas del frame."""
        nodes = self._detect_nodes(frame)
        roi = self.minigame_roi
        priority = {"green": 0, "neutral": 1}
        nodes.sort(key=lambda node: (priority[node.color], -node.brightness, node.point.y, node.point.x))
        return [
            Point(node.point.x + roi.x, node.point.y + roi.y)
            for node in nodes
        ]

    def find_start_node(self, frame: np.ndarray) -> Point | None:
        """Detecta el centro del nodo verde inicial en coordenadas absolutas."""
        green_nodes = [node for node in self._detect_nodes(frame) if node.color == "green"]
        if not green_nodes:
            return None
        node = max(green_nodes, key=lambda candidate: candidate.radius)
        return Point(
            node.point.x + self.minigame_roi.x,
            node.point.y + self.minigame_roi.y,
        )

    @staticmethod
    def _connected_nodes(
        crop: np.ndarray, nodes: list[_Node]
    ) -> dict[int, set[int]]:
        adjacency = {idx: set() for idx in range(len(nodes))}
        if len(nodes) < 2:
            return adjacency

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        height, width = crop.shape[:2]
        max_distance = float(np.hypot(width, height))
        for first in range(len(nodes)):
            for second in range(first + 1, len(nodes)):
                a = nodes[first]
                b = nodes[second]
                dx = b.point.x - a.point.x
                dy = b.point.y - a.point.y
                distance = float(np.hypot(dx, dy))
                if distance <= a.radius + b.radius or distance > max_distance:
                    continue
                margin = min(0.35, max(0.08, (a.radius + b.radius) * 0.65 / distance))
                offset = max(8, min(32, int(min(a.radius, b.radius) * 0.60)))
                sample_count = max(10, int(distance / 2.0))
                samples = np.linspace(margin, 1.0 - margin, sample_count)
                hit_count = 0
                valid_samples = 0
                for fraction in samples:
                    x = int(round(a.point.x + dx * fraction))
                    y = int(round(a.point.y + dy * fraction))
                    perpendicular_x = -dy / distance * offset
                    perpendicular_y = dx / distance * offset
                    side_a_x = int(round(x + perpendicular_x))
                    side_a_y = int(round(y + perpendicular_y))
                    side_b_x = int(round(x - perpendicular_x))
                    side_b_y = int(round(y - perpendicular_y))
                    if not (
                        0 <= x < width
                        and 0 <= y < height
                        and 0 <= side_a_x < width
                        and 0 <= side_b_x < width
                        and 0 <= side_a_y < height
                        and 0 <= side_b_y < height
                    ):
                        continue
                    valid_samples += 1

                    hue, saturation, value = hsv[y, x]
                    green_line = (
                        45 <= hue <= 85 and saturation >= 110 and value >= 120
                    )
                    center_brightness = gray[y, x]
                    side_brightness = (
                        gray[side_a_y, side_a_x] + gray[side_b_y, side_b_x]
                    ) / 2.0
                    neutral_line = (
                        saturation <= 90
                        and value >= 55
                        and center_brightness - side_brightness >= 14.0
                    )
                    if green_line or neutral_line:
                        hit_count += 1
                if valid_samples > 0 and hit_count / valid_samples >= 0.45:
                    adjacency[first].add(second)
                    adjacency[second].add(first)
        return adjacency

    @staticmethod
    def _order_nodes(nodes: list[_Node], adjacency: dict[int, set[int]]) -> list[int]:
        """Recorre solo los nodos conectados por líneas a partir del nodo verde inicial."""
        if not nodes:
            return []

        green_indices = [idx for idx, node in enumerate(nodes) if node.color == "green"]
        if not green_indices:
            return []
        start = max(green_indices, key=lambda idx: nodes[idx].radius)

        ordered = [start]
        visited = {start}

        def walk(current: int) -> None:
            neighbors = sorted(
                (idx for idx in adjacency[current] if idx not in visited),
                key=lambda idx: (
                    -nodes[idx].brightness,
                    float(
                        np.hypot(
                            nodes[idx].point.x - nodes[current].point.x,
                            nodes[idx].point.y - nodes[current].point.y,
                        )
                    ),
                ),
            )
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                ordered.append(neighbor)
                walk(neighbor)

        walk(start)
        return ordered

    def get_ordered_nodes(self, frame: np.ndarray) -> list[Point]:
        """Devuelve el recorrido conectado desde el nodo verde inicial hasta el nodo oscuro final."""
        crop = self._crop_roi(frame)
        if crop is None or self._find_bar(frame) is None:
            return []

        nodes = self._detect_nodes(frame)
        if not any(node.color == "green" for node in nodes):
            return []
        adjacency = self._connected_nodes(crop, nodes)
        indices = self._order_nodes(nodes, adjacency)
        roi = self.minigame_roi
        return [
            Point(nodes[idx].point.x + roi.x, nodes[idx].point.y + roi.y)
            for idx in indices
        ]

    def is_minigame_active(self, frame: np.ndarray) -> bool:
        """Exige barra superior y al menos un nodo verde para evitar falsos positivos."""
        return self._find_bar(frame) is not None and self.find_start_node(frame) is not None

    def analyze(self, frame: np.ndarray) -> MinigameState:
        """Ejecuta el análisis completo y devuelve estado vacío ante evidencia insuficiente."""
        bar_ratio = self.detect_time_bar(frame)
        start_node = self.find_start_node(frame)
        active = bar_ratio is not None and start_node is not None
        if not active:
            return MinigameState(active=False)
        return MinigameState(
            active=True,
            bar_ratio=bar_ratio,
            start_node=start_node,
            ordered_nodes=tuple(self.get_ordered_nodes(frame)),
        )
