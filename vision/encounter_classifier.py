"""Clasificador de encuentros, UI, formas y shards mediante template matching y color."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import cv2
import numpy as np

from core.types import EncounterResult, Form, ROI

logger = logging.getLogger(__name__)

BATTLE_MENU_TEMPLATES = ("fight", "items", "run", "loomians")
FISHING_RETURN_TEMPLATES = ("fishing_return", "fishing", "got_away_safely")
SHARD_STAT_WORDS = ("health", "energy", "attack", "defense", "speed")
FORM_TEMPLATE_NAMES = {
    "normal": Form.NORMAL,
    "corrupto": Form.CORRUPTO,
    "corrupted": Form.CORRUPTO,
    "alpha": Form.ALPHA,
    "gamma": Form.GAMMA,
}


class EncounterClassifier:
    """Clasifica retorno a pesca, menús, nombres, formas y shards sin OCR.

    - Nombres de Loomian, menú de combate, banners de corrupto y palabras de shard
      se detectan mediante template matching con OpenCV desde ``templates_dir``.
    - La forma ``normal`` exige evidencia positiva (paleta coincidente con referencia
      de especie y ausencia de señales de corrupto, destellos y wisp).
    """

    def __init__(
        self,
        templates_dir: str | Path,
        window_size: tuple[int, int],
        rois: Mapping[str, ROI],
        template_match_threshold: float = 0.85,
        min_confidence: float = 0.80,
        alpha_sparkle_hsv_lower: Sequence[int] = (92, 65, 165),
        alpha_sparkle_hsv_upper: Sequence[int] = (108, 165, 255),
        gamma_sparkle_hsv_lower: Sequence[int] = (72, 65, 165),
        gamma_sparkle_hsv_upper: Sequence[int] = (85, 220, 255),
        sparkle_min_pixels: int = 45,
        normal_palette_min_score: float = 0.78,
    ) -> None:
        self.templates_dir = Path(templates_dir)
        self.expected_width, self.expected_height = window_size
        self.rois: dict[str, ROI] = dict(rois)
        self.template_match_threshold = float(template_match_threshold)
        self.min_confidence = float(min_confidence)
        self.alpha_sparkle_lower = np.array(alpha_sparkle_hsv_lower, dtype=np.uint8)
        self.alpha_sparkle_upper = np.array(alpha_sparkle_hsv_upper, dtype=np.uint8)
        self.gamma_sparkle_lower = np.array(gamma_sparkle_hsv_lower, dtype=np.uint8)
        self.gamma_sparkle_upper = np.array(gamma_sparkle_hsv_upper, dtype=np.uint8)
        self.sparkle_min_pixels = int(sparkle_min_pixels)
        self.normal_palette_min_score = float(normal_palette_min_score)

        self._ui_templates: dict[str, np.ndarray] = {}
        self._loomian_templates: dict[str, np.ndarray] = {}
        self._form_templates: dict[str, np.ndarray] = {}
        self._shard_templates: dict[str, np.ndarray] = {}
        self._palette_references: dict[str, np.ndarray] = {}
        self._load_templates()

    def _load_templates(self) -> None:
        """Carga templates de UI, nombres, formas, paletas normales y shards."""
        self._ui_templates = self._load_dir(self.templates_dir / "ui")
        self._loomian_templates = self._load_dir(self.templates_dir / "loomians")
        self._form_templates = self._load_dir(self.templates_dir / "forms")
        self._shard_templates = self._load_dir(self.templates_dir / "shards")
        self._palette_references = self._load_palettes(self.templates_dir / "forms" / "palettes")
        logger.debug(
            "Plantillas cargadas desde %s: %d UI, %d Loomians, %d formas, %d paletas, %d shards.",
            self.templates_dir,
            len(self._ui_templates),
            len(self._loomian_templates),
            len(self._form_templates),
            len(self._palette_references),
            len(self._shard_templates),
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

    @staticmethod
    def _load_palettes(directory: Path) -> dict[str, np.ndarray]:
        palettes: dict[str, np.ndarray] = {}
        if not directory.is_dir():
            return palettes
        for path in sorted(directory.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                hist = np.asarray(data["histogram"], dtype=np.float32)
                if hist.ndim == 2 and hist.sum() > 0:
                    palettes[path.stem.casefold()] = hist
            except Exception as exc:
                logger.warning("No se pudo cargar la paleta normal %s: %s", path, exc)
        return palettes

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
    def _match_template_with_loc(
        region: np.ndarray,
        template: np.ndarray,
    ) -> tuple[float, tuple[int, int]]:
        """Ejecuta template matching normalizado y devuelve ``(score, (x, y))``."""
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
            return 0.0, (0, 0)
        if float(np.std(template)) < 1.0 or float(np.std(region)) < 1.0:
            return 0.0, (0, 0)
        result = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
        _, max_value, _, max_loc = cv2.minMaxLoc(result)
        if not np.isfinite(max_value):
            return 0.0, (0, 0)
        return float(max_value), (int(max_loc[0]), int(max_loc[1]))

    @classmethod
    def _match_template(cls, region: np.ndarray, template: np.ndarray) -> float:
        score, _ = cls._match_template_with_loc(region, template)
        return score

    @property
    def _required_confidence(self) -> float:
        return max(self.template_match_threshold, self.min_confidence)

    # ------------------------------------------------------------------
    # Detección de UI y banners
    # ------------------------------------------------------------------

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
        """Detecta el botón gris ``Cancel`` dentro del submenú de ``Fight``."""
        region = self._crop_roi(frame, "battle_menu")
        template = self._ui_templates.get("cancel")
        if region is None or template is None:
            return False
        return self._match_template(region, template) >= self._required_confidence

    def detect_fishing_return(self, frame: np.ndarray) -> bool:
        """Detecta el retorno a pesca con ``ui/fishing_return.png`` o ``ui/got_away_safely.png``."""
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

    def detect_corrupt_powerful_banner(self, frame: np.ndarray) -> bool:
        """Detecta el banner ``It looks more powerful than normal!`` en ``battle_banner``."""
        region = self._crop_roi(frame, "battle_banner")
        template = self._ui_templates.get("corrupt_powerful")
        if region is None or template is None:
            return False
        return self._match_template(region, template) >= self._required_confidence

    def detect_corrupt_appeared_banner(self, frame: np.ndarray) -> bool:
        """Detecta el prefijo ``A corrupt`` del banner de aparición en ``battle_banner``."""
        region = self._crop_roi(frame, "battle_banner")
        template = self._ui_templates.get("corrupt_appeared")
        if region is None or template is None:
            return False
        return self._match_template(region, template) >= self._required_confidence

    def detect_corrupt_card(self, frame: np.ndarray) -> bool:
        """Detecta si la tarjeta del enemigo en ``enemy_card`` es violeta (corrupto)."""
        region = self._crop_roi(frame, "enemy_card")
        if region is None:
            return False
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        purple_mask = cv2.inRange(hsv, np.array((125, 85, 80)), np.array((162, 255, 255)))
        cyan_mask = cv2.inRange(hsv, np.array((88, 75, 95)), np.array((112, 255, 255)))
        purple_pixels = int(np.count_nonzero(purple_mask))
        cyan_pixels = int(np.count_nonzero(cyan_mask))
        return purple_pixels >= 3500 and purple_pixels > cyan_pixels * 1.5

    # ------------------------------------------------------------------
    # Destellos, wisp y paletas normales por especie
    # ------------------------------------------------------------------

    def detect_entry_sparkle(self, frame: np.ndarray) -> Form | None:
        """Detecta destellos de entrada alpha (celeste) o gamma (verde) en ``loomian_sprite``.

        Devuelve ``Form.ALPHA``, ``Form.GAMMA``, ``Form.UNKNOWN`` (si ambos se activan)
        o ``None`` si no hay destello por encima del umbral configurado.
        """
        region = self._crop_roi(frame, "loomian_sprite")
        if region is None:
            return None
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        alpha_pixels = int(
            np.count_nonzero(cv2.inRange(hsv, self.alpha_sparkle_lower, self.alpha_sparkle_upper))
        )
        gamma_pixels = int(
            np.count_nonzero(cv2.inRange(hsv, self.gamma_sparkle_lower, self.gamma_sparkle_upper))
        )
        alpha_hit = alpha_pixels >= self.sparkle_min_pixels
        gamma_hit = gamma_pixels >= self.sparkle_min_pixels
        if alpha_hit and gamma_hit:
            return Form.UNKNOWN
        if alpha_hit:
            return Form.ALPHA
        if gamma_hit:
            return Form.GAMMA
        return None

    @staticmethod
    def detect_wisp_motion(
        prev_sprite_crop: np.ndarray,
        curr_sprite_crop: np.ndarray,
        min_moving_pixels: int = 80,
    ) -> bool:
        """Detecta movimiento compacto alrededor del sprite (indicativo de wisp gamma)."""
        if (
            prev_sprite_crop.shape != curr_sprite_crop.shape
            or prev_sprite_crop.ndim != 3
            or prev_sprite_crop.size == 0
        ):
            return False
        diff = cv2.absdiff(prev_sprite_crop, curr_sprite_crop)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, motion_mask = cv2.threshold(gray, 35, 255, cv2.THRESH_BINARY)
        moving_pixels = int(np.count_nonzero(motion_mask))
        return moving_pixels >= min_moving_pixels

    def detect_wisp(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        min_moving_pixels: int = 80,
    ) -> bool:
        """Compara el ROI ``loomian_sprite`` entre dos frames consecutivos para detectar un wisp."""
        prev_crop = self._crop_roi(prev_frame, "loomian_sprite")
        curr_crop = self._crop_roi(curr_frame, "loomian_sprite")
        if prev_crop is None or curr_crop is None:
            return False
        return self.detect_wisp_motion(prev_crop, curr_crop, min_moving_pixels=min_moving_pixels)

    @staticmethod
    def extract_sprite_palette(sprite_crop: np.ndarray) -> np.ndarray | None:
        """Extrae un histograma HSV 2D normalizado sobre píxeles vívidos del sprite."""
        if sprite_crop.ndim != 3 or sprite_crop.size == 0:
            return None
        hsv = cv2.cvtColor(sprite_crop, cv2.COLOR_BGR2HSV)
        # S>=100, V>=100 excluye roca gris y pasto verde oscuro del fondo en Grano Cave
        vivid_mask = cv2.inRange(hsv, np.array((0, 100, 100)), np.array((180, 255, 255)))
        if int(np.count_nonzero(vivid_mask)) < 150:
            return None
        hist = cv2.calcHist([hsv], [0, 1], vivid_mask, [30, 8], [0, 180, 0, 256])
        total = float(hist.sum())
        if total <= 0:
            return None
        return (hist / total).astype(np.float32)

    def match_normal_palette(self, frame: np.ndarray, species_name: str) -> float:
        """Compara la paleta actual de ``loomian_sprite`` contra la referencia normal."""
        ref = self._palette_references.get(species_name.strip().casefold())
        if ref is None:
            return 0.0
        region = self._crop_roi(frame, "loomian_sprite")
        if region is None:
            return 0.0
        current = self.extract_sprite_palette(region)
        if current is None:
            return 0.0
        score = float(cv2.compareHist(ref, current, cv2.HISTCMP_CORREL))
        return float(np.clip(score, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Lectura de Shards (Opción A: template matching por palabra + M/R)
    # ------------------------------------------------------------------

    def detect_shard_drop(self, frame: np.ndarray) -> str | None:
        """Detecta un banner de drop de shard y lee el tipo por palabra y glifo M/R.

        Si confirma el banner de drop (``Shard`` / ``dropped a`` / ``shard_drop``) pero
        no puede leer la palabra o el glifo ``M``/``R``, devuelve ``"Unknown"``.
        """
        region = self._crop_roi(frame, "battle_banner")
        if region is None:
            return None

        shard_word_tpl = self._ui_templates.get("shard_word")
        if shard_word_tpl is None:
            shard_word_tpl = self._shard_templates.get("shard")
        shard_score, shard_loc = (
            self._match_template_with_loc(region, shard_word_tpl)
            if shard_word_tpl is not None
            else (0.0, (0, 0))
        )
        shard_word_hit = shard_score >= self._required_confidence

        dropped_tpl = self._ui_templates.get("dropped_a")
        dropped_hit = (
            dropped_tpl is not None
            and self._match_template(region, dropped_tpl) >= self._required_confidence
        )
        generic_tpl = self._ui_templates.get("shard_drop")
        generic_hit = (
            generic_tpl is not None
            and self._match_template(region, generic_tpl) >= self._required_confidence
        )

        # 1. Soporte a templates compuestos directos (p. ej. shards/energy_shard.png)
        best_direct: str | None = None
        best_direct_score = 0.0
        for key, tpl in self._shard_templates.items():
            if key in SHARD_STAT_WORDS or key in {"m", "r", "shard"}:
                continue
            score = self._match_template(region, tpl)
            if score > best_direct_score:
                best_direct_score = score
                best_direct = key
        if best_direct is not None and best_direct_score >= self._required_confidence:
            return " ".join(part.capitalize() for part in best_direct.replace("_", " ").split())

        # 2. Lectura por palabra del stat en battle_banner
        best_stat: str | None = None
        best_stat_score = 0.0
        best_stat_loc = (0, 0)
        best_stat_size = (0, 0)
        for stat_word in SHARD_STAT_WORDS:
            tpl = self._shard_templates.get(stat_word)
            if tpl is None:
                continue
            score, loc = self._match_template_with_loc(region, tpl)
            if score > best_stat_score:
                best_stat_score = score
                best_stat = stat_word
                best_stat_loc = loc
                best_stat_size = (tpl.shape[1], tpl.shape[0])

        drop_confirmed = shard_word_hit or dropped_hit or generic_hit or (
            best_stat is not None and best_stat_score >= self._required_confidence
        )
        if not drop_confirmed:
            return None

        if best_stat is None or best_stat_score < self._required_confidence:
            return "Shard" if generic_hit and not (shard_word_hit or dropped_hit) else "Unknown"

        stat_label = best_stat.capitalize()
        if best_stat in {"attack", "defense"}:
            suffix = self._match_mr_suffix(
                region=region,
                stat_loc=best_stat_loc,
                stat_size=best_stat_size,
                shard_loc=shard_loc if shard_word_hit else None,
            )
            if suffix is None:
                return "Unknown"
            return f"{stat_label} {suffix} Shard"

        return f"{stat_label} Shard"

    def _match_mr_suffix(
        self,
        region: np.ndarray,
        stat_loc: tuple[int, int],
        stat_size: tuple[int, int],
        shard_loc: tuple[int, int] | None,
    ) -> str | None:
        """Busca el glifo ``M`` o ``R`` a la derecha de Attack/Defense o al inicio de la línea 2."""
        m_tpl = self._shard_templates.get("m")
        r_tpl = self._shard_templates.get("r")
        if m_tpl is None or r_tpl is None:
            return None

        reg_h, reg_w = region.shape[:2]
        word_x, word_y = stat_loc
        word_w, word_h = stat_size
        glyph_w = max(m_tpl.shape[1], r_tpl.shape[1])
        glyph_h = max(m_tpl.shape[0], r_tpl.shape[0])

        # Franja inmediata a la derecha de la palabra Attack/Defense
        candidates: list[np.ndarray] = []
        rx1 = min(reg_w, word_x + word_w)
        rx2 = min(reg_w, rx1 + max(45, glyph_w * 3))
        ry1 = max(0, word_y - 10)
        ry2 = min(reg_h, word_y + max(word_h, glyph_h) + 12)
        if rx2 - rx1 >= glyph_w and ry2 - ry1 >= glyph_h:
            candidates.append(region[ry1:ry2, rx1:rx2])

        # Si se partió la línea después de Attack/Defense: inicio de la línea 2 o izquierda de "Shard"
        if shard_loc is not None and shard_loc[1] > word_y + word_h // 2:
            sx, sy = shard_loc
            lx1 = max(0, sx - max(45, glyph_w * 3))
            lx2 = sx
            ly1 = max(0, sy - 10)
            ly2 = min(reg_h, sy + glyph_h + 12)
            if lx2 - lx1 >= glyph_w and ly2 - ly1 >= glyph_h:
                candidates.append(region[ly1:ly2, lx1:lx2])
        else:
            ny1 = min(reg_h, word_y + word_h // 2)
            if reg_h - ny1 >= glyph_h:
                candidates.append(region[ny1:reg_h, : min(reg_w, max(160, glyph_w * 6))])

        best_m = 0.0
        best_r = 0.0
        for patch in candidates:
            best_m = max(best_m, self._match_template(patch, m_tpl))
            best_r = max(best_r, self._match_template(patch, r_tpl))

        if max(best_m, best_r) < self._required_confidence:
            return None
        if abs(best_m - best_r) < 0.03:
            return None
        return "M" if best_m > best_r else "R"

    # ------------------------------------------------------------------
    # Clasificación de nombre, forma y encuentro completo
    # ------------------------------------------------------------------

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

    def classify_form(
        self,
        frame: np.ndarray,
        species_name: str | None = None,
    ) -> tuple[Form, float]:
        """Clasifica la forma exigiendo evidencia positiva; ante la duda devuelve UNKNOWN."""
        region = self._crop_roi(frame, "loomian_sprite")
        if region is None:
            return Form.UNKNOWN, 0.0

        # 1. Templates explícitos de forma (si existen en assets/templates/forms/*.png)
        if self._form_templates:
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
                best_form != Form.UNKNOWN
                and best_score >= self._required_confidence
                and not (
                    second_best_score >= self._required_confidence
                    and best_score - second_best_score < 0.03
                )
            ):
                return best_form, best_score
            return Form.UNKNOWN, min(best_score, max(0.0, self.min_confidence - 0.01))

        # 2. Señales inmediatas de forma especial visibles en el frame actual
        if self.detect_corrupt_card(frame) or self.detect_corrupt_powerful_banner(frame):
            return Form.CORRUPTO, 1.0

        sparkle = self.detect_entry_sparkle(frame)
        if sparkle in (Form.ALPHA, Form.GAMMA):
            return sparkle, 1.0
        if sparkle == Form.UNKNOWN:
            return Form.UNKNOWN, 0.0

        # 3. Evidencia positiva para forma normal mediante paleta por especie
        if species_name is not None:
            palette_score = self.match_normal_palette(frame, species_name)
            if palette_score >= self.normal_palette_min_score and palette_score >= self.min_confidence:
                return Form.NORMAL, palette_score
            return Form.UNKNOWN, min(palette_score, max(0.0, self.min_confidence - 0.01))

        return Form.UNKNOWN, 0.0

    def classify_encounter(self, frame: np.ndarray) -> EncounterResult:
        """Clasifica un encuentro únicamente si menú, nombre y forma son concluyentes."""
        if not self.is_valid_frame_size(frame):
            return EncounterResult(
                name=None, form=Form.UNKNOWN, confidence=0.0, item_detected=False
            )

        if not self.detect_battle_menu(frame):
            return EncounterResult(
                name=None, form=Form.UNKNOWN, confidence=0.0, item_detected=False
            )

        name, name_confidence = self.classify_name(frame)
        form, form_confidence = self.classify_form(frame, species_name=name)
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
