"""Punto de entrada del bot de pesca para Loomian Legacy."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

from alerts.sound_alert import SoundAlert
from core.controller import BotController, RunListError
from core.state_machine import FishingStateMachine
from core.types import Point, ROI, State
from debug.visualizer import Visualizer
from vision.encounter_classifier import EncounterClassifier
from vision.minigame_tracker import MinigameTracker
from vision.screen_capture import ScreenCapture

logger = logging.getLogger(__name__)


class KillSwitchAdapter(Protocol):
    """Interfaz del adaptador de kill-switch global para permitir fakes en tests."""

    def register(self) -> None:
        """Registra el hook global de teclado."""

    def unregister(self) -> None:
        """Libera el hook global de teclado."""

    def is_triggered(self) -> bool:
        """Devuelve ``True`` si el kill-switch fue activado."""

    def trigger(self) -> None:
        """Activa el kill-switch manualmente."""


class KillSwitchRegistrationError(RuntimeError):
    """Error lanzado cuando no se puede registrar el hook global del kill-switch."""


class KeyboardKillSwitch:
    """Adaptador de kill-switch global sobre la librería ``keyboard``."""

    def __init__(self, key: str = "F8") -> None:
        self.key = key
        self._event = threading.Event()
        self._hotkey_handle: Any = None
        self._keyboard_module: Any = None

    def register(self) -> None:
        """Registra el atajo global; lanza ``KillSwitchRegistrationError`` si falla."""
        try:
            import keyboard

            self._keyboard_module = keyboard
            self._hotkey_handle = keyboard.add_hotkey(self.key, self.trigger)
            logger.info("Kill-switch global registrado en la tecla '%s'.", self.key)
        except Exception as exc:
            raise KillSwitchRegistrationError(
                f"No se pudo registrar el kill-switch global en '{self.key}': {exc}"
            ) from exc

    def unregister(self) -> None:
        """Elimina el atajo global si estaba registrado."""
        if self._keyboard_module is not None and self._hotkey_handle is not None:
            try:
                self._keyboard_module.remove_hotkey(self._hotkey_handle)
            except Exception as exc:
                logger.debug("Error al desregistrar kill-switch: %s", exc)
            finally:
                self._hotkey_handle = None

    def trigger(self) -> None:
        """Marca el kill-switch como activado."""
        if not self._event.is_set():
            logger.warning("Kill-switch (%s) activado.", self.key)
            self._event.set()

    def is_triggered(self) -> bool:
        """Indica si el kill-switch fue activado."""
        return self._event.is_set()


def setup_kill_switch(kill_switch: KillSwitchAdapter, dry_run: bool) -> bool:
    """Registra el kill-switch respetando la política de permisos en dry-run vs. real.

    - En ``dry_run=True``: emite warning claro y devuelve ``True`` para continuar.
    - En ``dry_run=False``: emite error indicando permisos de administrador y devuelve ``False``.
    """
    try:
        kill_switch.register()
        return True
    except KillSwitchRegistrationError as exc:
        if dry_run:
            logger.warning(
                "No se pudo registrar el kill-switch en modo dry-run (%s). "
                "Se continúa solo porque dry-run no ejecuta clics ni teclas reales.",
                exc,
            )
            return True

        logger.error(
            "ABORTADO antes de arrancar en modo real: %s. "
            "En Windows, registrar hooks globales con 'keyboard' puede requerir "
            "ejecutar la terminal como Administrador.",
            exc,
        )
        return False


class ConfigError(ValueError):
    """Error legible de configuración detectado al iniciar el bot."""


REQUIRED_ROIS = ("minigame", "battle_menu", "loomian_name", "loomian_sprite", "battle_banner", "enemy_card")
REQUIRED_UI_POINTS = ("cast_button", "fight_button", "run_button", "cancel_button", "corrupt_move_slot")
REQUIRED_TIMEOUTS = (
    "IDLE_OR_CAST",
    "WAITING_MINIGAME",
    "SOLVING_DOTS",
    "RESOLVING_OUTCOME",
    "IN_BATTLE",
    "COOLDOWN",
)


def load_settings(path: str | Path) -> dict[str, Any]:
    """Carga ``config/settings.json``; lanza ``ConfigError`` con mensaje legible si falla."""
    settings_path = Path(path)
    try:
        with settings_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError as exc:
        raise ConfigError(f"No se encontró {settings_path}.") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"{settings_path} no es JSON válido (línea {exc.lineno}, columna {exc.colno}): {exc.msg}"
        ) from exc
    if not isinstance(data, dict):
        raise ConfigError(f"El archivo {settings_path} debe contener un objeto JSON.")
    return data


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_settings(settings: dict[str, Any]) -> list[str]:
    """Valida claves obligatorias, tipos y límites de ``settings.json``.

    Devuelve la lista de errores encontrados (vacía si la configuración es válida).
    """
    errors: list[str] = []

    window = settings.get("window_size")
    width = height = 0
    if not isinstance(window, dict) or not _is_int(window.get("width")) or not _is_int(window.get("height")):
        errors.append("'window_size' debe tener 'width' y 'height' enteros.")
    else:
        width, height = window["width"], window["height"]
        if width <= 0 or height <= 0:
            errors.append("'window_size.width' y 'window_size.height' deben ser > 0.")

    origin = settings.get("window_origin")
    if origin is not None and (
        not isinstance(origin, dict) or not _is_int(origin.get("x")) or not _is_int(origin.get("y"))
    ):
        errors.append("'window_origin' debe tener 'x' e 'y' enteros.")

    rois = settings.get("rois")
    if not isinstance(rois, dict):
        errors.append("Falta la sección 'rois'.")
    else:
        for name in REQUIRED_ROIS:
            if name not in rois:
                errors.append(f"Falta el ROI obligatorio 'rois.{name}'.")
        for name, cfg in rois.items():
            if not isinstance(cfg, dict) or not all(_is_int(cfg.get(k)) for k in ("x", "y", "width", "height")):
                errors.append(f"'rois.{name}' debe tener 'x', 'y', 'width' y 'height' enteros.")
                continue
            if cfg["width"] <= 0 or cfg["height"] <= 0:
                errors.append(f"'rois.{name}' debe tener ancho y alto > 0.")
            elif width > 0 and height > 0 and (
                cfg["x"] < 0
                or cfg["y"] < 0
                or cfg["x"] + cfg["width"] > width
                or cfg["y"] + cfg["height"] > height
            ):
                errors.append(f"'rois.{name}' queda fuera de window_size ({width}x{height}).")

    points = settings.get("ui_points")
    if not isinstance(points, dict):
        errors.append("Falta la sección 'ui_points'.")
    else:
        for name in REQUIRED_UI_POINTS:
            if name not in points:
                errors.append(f"Falta la coordenada obligatoria 'ui_points.{name}'.")
        for name, cfg in points.items():
            if not isinstance(cfg, dict) or not _is_int(cfg.get("x")) or not _is_int(cfg.get("y")):
                errors.append(f"'ui_points.{name}' debe tener 'x' e 'y' enteros.")
            elif width > 0 and height > 0 and not (0 <= cfg["x"] < width and 0 <= cfg["y"] < height):
                errors.append(f"'ui_points.{name}' queda fuera de window_size ({width}x{height}).")

    timeouts = settings.get("timeouts")
    if not isinstance(timeouts, dict):
        errors.append("Falta la sección 'timeouts'.")
    else:
        for name in REQUIRED_TIMEOUTS:
            value = timeouts.get(name)
            if not _is_number(value) or value <= 0:
                errors.append(f"'timeouts.{name}' debe ser un número > 0.")

    rate = settings.get("capture_rate_limit")
    if not isinstance(rate, dict) or not all(
        _is_number(rate.get(k)) and rate.get(k) > 0
        for k in ("idle_fps", "resolving_outcome_fps", "solving_dots_fps")
    ):
        errors.append(
            "'capture_rate_limit' debe tener 'idle_fps', 'resolving_outcome_fps' y 'solving_dots_fps' > 0."
        )

    if not isinstance(settings.get("kill_switch_key"), str) or not settings.get("kill_switch_key"):
        errors.append("'kill_switch_key' debe ser un texto no vacío (ej. \"F8\").")
    if not isinstance(settings.get("stop_after_corrupt"), bool):
        errors.append("'stop_after_corrupt' debe ser true o false.")
    for key in ("template_match_threshold", "min_confidence"):
        value = settings.get(key)
        if not _is_number(value) or not 0.0 <= value <= 1.0:
            errors.append(f"'{key}' debe ser un número entre 0.0 y 1.0.")
    if not isinstance(settings.get("templates_dir"), str):
        errors.append("'templates_dir' debe ser una ruta (texto).")

    paths = settings.get("paths")
    if not isinstance(paths, dict) or not all(
        isinstance(paths.get(k), str) and paths.get(k) for k in ("run_list", "corruptos_csv")
    ):
        errors.append("'paths' debe incluir 'run_list' y 'corruptos_csv' como rutas.")

    solving = settings.get("solving")
    if solving is not None:
        if not isinstance(solving, dict):
            errors.append("'solving' debe ser un objeto.")
        else:
            delay = solving.get("dot_move_delay_s", 0.0)
            if not _is_number(delay) or delay < 0:
                errors.append("'solving.dot_move_delay_s' debe ser un número >= 0.")
            ratio = solving.get("min_bar_ratio", 0.0)
            if not _is_number(ratio) or not 0.0 <= ratio <= 1.0:
                errors.append("'solving.min_bar_ratio' debe estar entre 0.0 y 1.0.")
            retries = solving.get("max_solving_retries", 1)
            if not _is_int(retries) or retries < 1:
                errors.append("'solving.max_solving_retries' debe ser un entero >= 1.")

    resolving = settings.get("resolving")
    if resolving is not None:
        if not isinstance(resolving, dict):
            errors.append("'resolving' debe ser un objeto.")
        else:
            confirm_s = resolving.get("fishing_return_confirm_s", 0.0)
            if not _is_number(confirm_s) or confirm_s < 0:
                errors.append("'resolving.fishing_return_confirm_s' debe ser un número >= 0.")
            dark_thr = resolving.get("dark_frame_mean_threshold", 0.0)
            if not _is_number(dark_thr) or not 0.0 <= dark_thr <= 255.0:
                errors.append("'resolving.dark_frame_mean_threshold' debe estar entre 0 y 255.")

    form_cfg = settings.get("form_detection")
    if form_cfg is not None:
        if not isinstance(form_cfg, dict):
            errors.append("'form_detection' debe ser un objeto.")
        else:
            for hsv_key in (
                "alpha_sparkle_hsv_lower",
                "alpha_sparkle_hsv_upper",
                "gamma_sparkle_hsv_lower",
                "gamma_sparkle_hsv_upper",
            ):
                vec = form_cfg.get(hsv_key)
                if vec is not None and (
                    not isinstance(vec, list) or len(vec) != 3 or not all(_is_int(v) and 0 <= v <= 255 for v in vec)
                ):
                    errors.append(f"'form_detection.{hsv_key}' debe ser una lista de 3 enteros 0..255.")
            pal_min = form_cfg.get("normal_palette_min_score", 0.0)
            if not _is_number(pal_min) or not 0.0 <= pal_min <= 1.0:
                errors.append("'form_detection.normal_palette_min_score' debe estar entre 0.0 y 1.0.")
            confirm_n = form_cfg.get("form_confirm_frames", 1)
            if not _is_int(confirm_n) or confirm_n < 1:
                errors.append("'form_detection.form_confirm_frames' debe ser un entero >= 1.")

    return errors


def validate_run_list_file(path: str | Path) -> list[str]:
    """Valida ``run_list.json`` al inicio. Un archivo ausente es válido (lista vacía)."""
    run_list_path = Path(path)
    if not run_list_path.is_file():
        return []
    try:
        BotController(
            run_list_path=run_list_path,
            corruptos_csv_path=run_list_path.with_suffix(".unused.csv"),
            ui_points={},
            dry_run=True,
        ).load_run_list()
    except RunListError as exc:
        return [str(exc)]
    return []


def _parse_rois(raw_rois: dict[str, dict[str, int]]) -> dict[str, ROI]:
    return {
        name: ROI(
            x=int(cfg["x"]),
            y=int(cfg["y"]),
            width=int(cfg["width"]),
            height=int(cfg["height"]),
        )
        for name, cfg in raw_rois.items()
    }


def _parse_points(raw_points: dict[str, dict[str, int]]) -> dict[str, Point]:
    return {
        name: Point(x=int(cfg["x"]), y=int(cfg["y"]))
        for name, cfg in raw_points.items()
    }


def prompt_add_to_run_list(
    loomian_name: str,
    controller: BotController,
    input_fn: Callable[[str], str] = input,
) -> bool:
    """Ofrece por consola (con el loop ya detenido) agregar el Loomian a ``run_list.json``."""
    try:
        answer = input_fn(
            f"¿Deseás agregar '{loomian_name}' a config/run_list.json para huir en futuros encuentros? [s/N]: "
        )
    except EOFError:
        return False

    if answer.strip().casefold() in {"s", "si", "sí", "y", "yes"}:
        try:
            added = controller.add_to_run_list(loomian_name)
        except RunListError as exc:
            logger.error("No se pudo agregar '%s': %s", loomian_name, exc)
            return False
        if added:
            logger.info("'%s' fue agregado a run_list.json.", loomian_name)
        else:
            logger.info("'%s' ya figuraba en run_list.json.", loomian_name)
        return added
    return False


def build_arg_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos CLI."""
    parser = argparse.ArgumentParser(
        description="Bot de pesca para Loomian Legacy (dry-run por defecto)."
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Simula acciones sin ejecutar clics ni teclas reales (activo por defecto; usar --no-dry-run para modo real).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activa logging detallado y depuración visual.",
    )
    parser.add_argument(
        "--settings",
        default="config/settings.json",
        help="Ruta al archivo settings.json (por defecto: config/settings.json).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Límite opcional de iteraciones del loop principal (útil para pruebas).",
    )
    return parser


def configure_logging(debug: bool) -> None:
    """Configura el formato y nivel de logging."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada principal del programa."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging(debug=args.debug)

    dry_run: bool = args.dry_run
    mode_label = "DRY-RUN (sin clics ni teclas reales)" if dry_run else "MODO REAL"
    logger.info("Iniciando Fishing Bot en %s.", mode_label)

    try:
        settings = load_settings(args.settings)
    except ConfigError as exc:
        logger.error("Error de configuración: %s", exc)
        return 1

    config_errors = validate_settings(settings)
    if not config_errors:
        config_errors.extend(validate_run_list_file(settings["paths"]["run_list"]))
    if config_errors:
        logger.error("Configuración inválida; abortando antes de arrancar:")
        for err in config_errors:
            logger.error("  - %s", err)
        return 1

    rois = _parse_rois(settings["rois"])
    ui_points = _parse_points(settings["ui_points"])
    window_cfg = settings["window_size"]
    window_size = (int(window_cfg["width"]), int(window_cfg["height"]))
    origin_cfg = settings.get("window_origin", {"x": 0, "y": 0})
    window_origin = Point(x=int(origin_cfg["x"]), y=int(origin_cfg["y"]))
    capture_region = ROI(
        x=window_origin.x,
        y=window_origin.y,
        width=window_size[0],
        height=window_size[1],
    )
    rate_cfg = settings["capture_rate_limit"]
    paths_cfg = settings["paths"]

    kill_switch = KeyboardKillSwitch(key=str(settings.get("kill_switch_key", "F8")))
    if not setup_kill_switch(kill_switch, dry_run=dry_run):
        return 1

    form_cfg = settings.get("form_detection", {})
    resolving_cfg = settings.get("resolving", {})

    capture = ScreenCapture(
        idle_fps=float(rate_cfg["idle_fps"]),
        resolving_outcome_fps=float(rate_cfg["resolving_outcome_fps"]),
        solving_dots_fps=float(rate_cfg["solving_dots_fps"]),
        fallback_size=window_size if dry_run else None,
        default_roi=capture_region,
    )
    tracker = MinigameTracker(minigame_roi=rois["minigame"])
    classifier = EncounterClassifier(
        templates_dir=settings["templates_dir"],
        window_size=window_size,
        rois=rois,
        template_match_threshold=float(settings["template_match_threshold"]),
        min_confidence=float(settings["min_confidence"]),
        alpha_sparkle_hsv_lower=form_cfg.get("alpha_sparkle_hsv_lower", (92, 65, 165)),
        alpha_sparkle_hsv_upper=form_cfg.get("alpha_sparkle_hsv_upper", (108, 165, 255)),
        gamma_sparkle_hsv_lower=form_cfg.get("gamma_sparkle_hsv_lower", (72, 65, 165)),
        gamma_sparkle_hsv_upper=form_cfg.get("gamma_sparkle_hsv_upper", (85, 220, 255)),
        sparkle_min_pixels=int(form_cfg.get("sparkle_min_pixels", 45)),
        normal_palette_min_score=float(form_cfg.get("normal_palette_min_score", 0.78)),
    )
    controller = BotController(
        run_list_path=paths_cfg["run_list"],
        corruptos_csv_path=paths_cfg["corruptos_csv"],
        ui_points=ui_points,
        dry_run=dry_run,
        window_origin=window_origin,
    )
    alert = SoundAlert(wav_path=paths_cfg.get("alert_wav"))
    visualizer = Visualizer() if args.debug else None

    solving_cfg = settings.get("solving", {})
    state_machine = FishingStateMachine(
        capture=capture,
        tracker=tracker,
        classifier=classifier,
        controller=controller,
        alert=alert,
        kill_switch=kill_switch,
        timeouts=settings["timeouts"],
        min_confidence=float(settings["min_confidence"]),
        stop_after_corrupt=bool(settings.get("stop_after_corrupt", False)),
        dot_move_delay_s=float(solving_cfg.get("dot_move_delay_s", 0.0)),
        min_bar_ratio=float(solving_cfg.get("min_bar_ratio", 0.05)),
        max_solving_retries=int(solving_cfg.get("max_solving_retries", 5)),
        fishing_return_confirm_s=float(resolving_cfg.get("fishing_return_confirm_s", 2.0)),
        dark_frame_mean_threshold=float(resolving_cfg.get("dark_frame_mean_threshold", 25.0)),
        form_confirm_frames=int(form_cfg.get("form_confirm_frames", 3)),
        unknown_shards_dir=form_cfg.get("unknown_shards_dir", "debug/unknown_shards"),
    )

    steps = 0
    try:
        while not state_machine.is_terminal:
            state = state_machine.step()
            steps += 1

            if visualizer is not None and state_machine.last_frame is not None:
                _ = visualizer.render_debug_frame(
                    state_machine.last_frame,
                    rois=rois,
                    points=state_machine.last_points,
                    labels=[f"Estado: {state.value}", f"Paso: {steps}", f"Dry-run: {dry_run}"],
                )

            if args.max_steps is not None and steps >= args.max_steps:
                logger.info("Límite --max-steps (%d) alcanzado. Finalizando limpiamente.", args.max_steps)
                break

            if state == State.COOLDOWN:
                time.sleep(0.05)
    except KeyboardInterrupt:
        logger.info("Ejecución interrumpida por el usuario (Ctrl+C).")
    finally:
        kill_switch.unregister()
        capture.close()

    if (
        state_machine.state == State.STOPPED_ALERT
        and state_machine.pending_run_list_candidate
        and sys.stdin.isatty()
    ):
        prompt_add_to_run_list(state_machine.pending_run_list_candidate, controller)

    logger.info("Bot finalizado en estado %s tras %d paso(s).", state_machine.state.value, steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
