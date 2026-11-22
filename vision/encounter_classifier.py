"""Clasificador de encuentros y UI mediante template matching con OpenCV (sin OCR)."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np

from core.types import EncounterResult, Form, ROI

logger = logging.getLogger(__name__)

BATTLE_MENU_TEMPLATES = ("fight", "items", "run", "loomians")
FISHING_RETURN_TEMPLATES = ("fishing_return", "fishing", "got_away_safely")
FORM_TEMPLATE_NAMES = {
    "normal": Form.NORMAL,
    "corrupto": Form.CORRUPTO,
    "corrupted": Form.CORRUPTO,
    "alpha": Form.ALPHA,
    "gamma": Form.GAMMA,
}


class EncounterClassifier:
    """Clasifica avisos, menús y nombres usando templates PNG configurables.

    La forma del Loomian solo se clasifica si existe un template explícito en
    ``templates_dir/forms``. La ausencia de evidencia nunca se interpreta como normal.
    """

    def __init__(
        self,
        templates_dir: str | Path,
        window_size: tuple[int, int],
        rois: Mapping[str, ROI],
        template_match_threshold: float = 0.85,
        min_confidence: float = 0.80,
    ) -> None:
        self.templates_dir = Path(templates_dir)
        self.expected_width, self.expected_height = window_size
        self.rois: dict[str, ROI] = dict(rois)
        self.template_match_threshold = float(template_match_threshold)
        self.min_confidence = float(min_confidence)

        self._ui_templates: dict[str, np.ndarray] = {}
        self._loomian_templates: dict[str, np.ndarray] = {}
        self._form_templates: dict[str, np.ndarray] = {}
        self._load_templates()

    def _load_templates(self) -> None:
        """Carga templates de UI, nombres y formas desde subdirectorios opcionales."""
        self._ui_templates = self._load_dir(self.templates_dir / "ui")
        self._loomian_templates = self._load_dir(self.templates_dir / "loomians")
        self._form_templates = self._load_dir(self.templates_dir / "forms")
        logger.debug(
            "Plantillas cargadas desde %s: %d UI, %d Loomians, %d formas.",
            self.templates_dir,
            len(self._ui_templates),
            len(self._loomian_templates),
            len(self._form_templates),
        )

    @staticmethod
    def _load_dir(directory: Path) -> dict[str, np.ndarray]:
        templates: dict[str, np.ndarray] = {}
        if not directory.is_dir():
            return templates
        for path in sorted(directory.glob("*.png")):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is not None and image.size > 0:
                templates[path.stem.casefold()] = image
            else:
                logger.warning("No se pudo cargar el template de imagen: %s", path)
        return templates

    def is_valid_frame_size(self, frame: np.ndarray) -> bool:
        """Valida formato BGR y dimensiones completas configuradas para la ventana."""
        if (
            not isinstance(frame, np.ndarray)
            or frame.dtype != np.uint8
            or frame.ndim != 3
            or frame.shape[2] < 3
        ):
            return False
        height, width = frame.shape[:2]
        return width == self.expected_width and height == self.expected_height

    def _crop_roi(self, frame: np.ndarray, roi_name: str) -> np.ndarray | None:
        """Recorta un ROI solo si el frame tiene el tamaño esperado y el ROI es válido."""
        if not self.is_valid_frame_size(frame):
            shape = getattr(frame, "shape", None)
            received_dims = (
                f"{shape[1]}x{shape[0]}"
                if isinstance(shape, tuple) and len(shape) >= 2
                else "desconocido"
            )
            logger.warning(
                "Tamaño o formato de frame inválido: recibido %s (shape=%s), esperado %dx%d.",
                received_dims,
                shape,
                self.expected_width,
                self.expected_height,
            )
            return None

        roi = self.rois.get(roi_name)
        if roi is None or roi.width <= 0 or roi.height <= 0:
            return None

        x1, y1 = roi.x, roi.y
        x2, y2 = x1 + roi.width, y1 + roi.height
        if x1 < 0 or y1 < 0 or x2 > self.expected_width or y2 > self.expected_height:
            logger.warning(
                "ROI '%s' excede window_size (%dx%d).",
                roi_name,
                self.expected_width,
                self.expected_height,
            )
            return None
        region = frame[y1:y2, x1:x2, :3]
        return region if region.size > 0 else None

    @staticmethod
    def _match_template(region: np.ndarray, template: np.ndarray) -> float:
        """Ejecuta template matching normalizado y devuelve el score finito máximo."""
        if (
            not isinstance(region, np.ndarray)
            or not isinstance(template, np.ndarray)
            or region.dtype != np.uint8
            or template.dtype != np.uint8
            or region.ndim != 3
            or template.ndim != 3
            or region.shape[2] != template.shape[2]
            or region.shape[0] < template.shape[0]
            or region.shape[1] < template.shape[1]
        ):
            return 0.0
        if float(np.std(template)) < 1.0 or float(np.std(region)) < 1.0:
            return 0.0
        result = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
        _, max_value, _, _ = cv2.minMaxLoc(result)
        return float(max_value) if np.isfinite(max_value) else 0.0

    @property
    def _required_confidence(self) -> float:
        return max(self.template_match_threshold, self.min_confidence)

    def detect_item_notice(self, frame: np.ndarray) -> bool:
        """Detecta el aviso de ítem con el template ``ui/item_notice.png``."""
        region = self._crop_roi(frame, "item_notice")
        template = self._ui_templates.get("item_notice")
        if region is None or template is None:
            return False
        return self._match_template(region, template) >= self._required_confidence

    def detect_battle_menu(self, frame: np.ndarray) -> bool:
        """Exige coincidencias claras para los cuatro botones del menú de batalla."""
        region = self._crop_roi(frame, "battle_menu")
        if region is None or any(key not in self._ui_templates for key in BATTLE_MENU_TEMPLATES):
            return False

        scores = {
            key: self._match_template(region, self._ui_templates[key])
            for key in BATTLE_MENU_TEMPLATES
        }
        matched = all(score >= self._required_confidence for score in scores.values())
        if not matched and max(scores.values()) >= 0.50:
            logger.debug(
                "battle_menu no confirmado (umbral=%.2f): %s",
                self._required_confidence,
                {k: round(v, 3) for k, v in scores.items()},
            )
        return matched

    def detect_fight_cancel(self, frame: np.ndarray) -> bool:
        """Detecta el botón gris ``Cancel`` dentro del submenú de ``Fight`` (``ui/cancel.png``)."""
        region = self._crop_roi(frame, "battle_menu")
        template = self._ui_templates.get("cancel")
        if region is None or template is None:
            return False
        return self._match_template(region, template) >= self._required_confidence

    def detect_fishing_return(self, frame: np.ndarray) -> bool:
        """Detecta el retorno a pesca solo con un template explícito de UI.

        Se admite ``ui/fishing_return.png`` (o ``ui/fishing.png``). Si no hay un
        ROI dedicado configurado, la búsqueda se realiza sobre el frame completo.
        """
        if not self.is_valid_frame_size(frame):
            return False
        for key in FISHING_RETURN_TEMPLATES:
            template = self._ui_templates.get(key)
            if template is None:
                continue
            if "fishing_return" in self.rois:
                region = self._crop_roi(frame, "fishing_return")
                if region is None:
                    return False
            else:
                region = frame[:, :, :3]
            if self._match_template(region, template) >= self._required_confidence:
                return True
        return False

    @staticmethod
    def _canonical_name(template_name: str) -> str:
        known_names = {"arketta": "Arketta", "goppie": "Goppie"}
        return known_names.get(template_name.casefold(), template_name)

    def classify_name(self, frame: np.ndarray) -> tuple[str | None, float]:
        """Identifica el Loomian en ``loomian_name`` sin usar OCR."""
        region = self._crop_roi(frame, "loomian_name")
        if region is None or not self._loomian_templates:
            return None, 0.0

        best_name: str | None = None
        best_score = 0.0
        second_best_score = 0.0
        for template_name, template in self._loomian_templates.items():
            score = self._match_template(region, template)
            if score > best_score:
                second_best_score = best_score
                best_score = score
                best_name = template_name
            elif score > second_best_score:
                second_best_score = score

        if best_name is None or best_score < self._required_confidence:
            return None, best_score
        if (
            second_best_score >= self._required_confidence
            and best_score - second_best_score < 0.03
        ):
            return None, min(best_score, max(0.0, self.min_confidence - 0.01))
        return self._canonical_name(best_name), best_score

    def classify_form(self, frame: np.ndarray) -> tuple[Form, float]:
        """Clasifica la forma solo con un template explícito y confianza suficiente."""
        region = self._crop_roi(frame, "loomian_sprite")
        if region is None or not self._form_templates:
            return Form.UNKNOWN, 0.0

        best_form = Form.UNKNOWN
        best_score = 0.0
        second_best_score = 0.0
        for template_name, template in self._form_templates.items():
            form = FORM_TEMPLATE_NAMES.get(template_name)
            if form is None:
                logger.warning("Template de forma no reconocido: %s", template_name)
                continue
            score = self._match_template(region, template)
            if score > best_score:
                second_best_score = best_score
                best_form = form
                best_score = score
            elif score > second_best_score:
                second_best_score = score

        if (
            best_form == Form.UNKNOWN
            or best_score < self._required_confidence
            or (
                second_best_score >= self._required_confidence
                and best_score - second_best_score < 0.03
            )
        ):
            low_confidence = min(
                best_score,
                max(0.0, self.min_confidence - 0.01),
            )
            return Form.UNKNOWN, low_confidence
        return best_form, best_score

    def classify_encounter(self, frame: np.ndarray) -> EncounterResult:
        """Clasifica un encuentro únicamente si menú, nombre y forma son concluyentes."""
        if not self.is_valid_frame_size(frame):
            return EncounterResult(
                name=None, form=Form.UNKNOWN, confidence=0.0, item_detected=False
            )

        if self.detect_item_notice(frame):
            return EncounterResult(
                name=None, form=Form.UNKNOWN, confidence=1.0, item_detected=True
            )
        if not self.detect_battle_menu(frame):
            return EncounterResult(
                name=None, form=Form.UNKNOWN, confidence=0.0, item_detected=False
            )

        name, name_confidence = self.classify_name(frame)
        form, form_confidence = self.classify_form(frame)
        confidence = min(name_confidence, form_confidence)

        if (
            name is None
            or form == Form.UNKNOWN
            or confidence < self.min_confidence
        ):
            return EncounterResult(
                name=name,
                form=Form.UNKNOWN,
                confidence=min(confidence, max(0.0, self.min_confidence - 0.01)),
                item_detected=False,
            )

        return EncounterResult(
            name=name,
            form=form,
            confidence=confidence,
            item_detected=False,
        )
