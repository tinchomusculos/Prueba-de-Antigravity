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
from core.controller import BotController
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


def load_settings(path: str | Path) -> dict[str, Any]:
    """Carga ``config/settings.json``."""
    settings_path = Path(path)
    with settings_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"El archivo {settings_path} debe contener un objeto JSON.")
    return data


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
        added = controller.add_to_run_list(loomian_name)
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

    settings = load_settings(args.settings)
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

    capture = ScreenCapture(
        idle_fps=float(rate_cfg["idle_fps"]),
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
