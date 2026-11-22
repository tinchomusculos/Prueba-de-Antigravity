"""Controlador de entrada (mouse/teclado), run_list.json en vivo y registro CSV de corruptos."""

from __future__ import annotations

import csv
import json
import logging
import os
import tempfile
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from core.types import Point

logger = logging.getLogger(__name__)

CSV_HEADER = ("timestamp", "loomian", "shard")


class InputDriver(Protocol):
    """Interfaz para acciones de mouse y teclado (real o dry-run)."""

    def move_to(self, point: Point) -> None:
        """Mueve el cursor a la coordenada indicada."""

    def click(self, point: Point, button: str = "left") -> None:
        """Hace clic en la coordenada indicada."""

    def press_key(self, key: str) -> None:
        """Presiona una tecla."""


class DryRunInputDriver:
    """Implementación segura que solo registra las acciones sin mover mouse ni pulsar teclas."""

    def __init__(self) -> None:
        self.recorded_actions: list[tuple[str, Any]] = []

    def move_to(self, point: Point) -> None:
        self.recorded_actions.append(("move_to", point))
        logger.info("[DRY-RUN] move_to(%d, %d)", point.x, point.y)

    def click(self, point: Point, button: str = "left") -> None:
        self.recorded_actions.append(("click", (point, button)))
        logger.info("[DRY-RUN] click(%d, %d, button=%s)", point.x, point.y, button)

    def press_key(self, key: str) -> None:
        self.recorded_actions.append(("press_key", key))
        logger.info("[DRY-RUN] press_key(%r)", key)


class PyAutoGuiInputDriver:
    """Implementación real de mouse y teclado sobre ``pyautogui``."""

    def __init__(self) -> None:
        import pyautogui  # Importación diferida para no exigir entorno gráfico en tests/dry-run

        pyautogui.FAILSAFE = True
        self._pyautogui = pyautogui

    def move_to(self, point: Point) -> None:
        self._pyautogui.moveTo(point.x, point.y)

    def click(self, point: Point, button: str = "left") -> None:
        self._pyautogui.click(x=point.x, y=point.y, button=button)

    def press_key(self, key: str) -> None:
        self._pyautogui.press(key)


class BotController:
    """Coordina acciones de entrada, lectura/escritura de run_list.json y log de corruptos."""

    def __init__(
        self,
        run_list_path: str | Path,
        corruptos_csv_path: str | Path,
        ui_points: Mapping[str, Point],
        input_driver: InputDriver | None = None,
        dry_run: bool = True,
        now_provider: Callable[[], datetime] | None = None,
        window_origin: Point = Point(0, 0),
    ) -> None:
        self.run_list_path = Path(run_list_path)
        self.corruptos_csv_path = Path(corruptos_csv_path)
        self.ui_points: dict[str, Point] = dict(ui_points)
        self.dry_run = dry_run
        self._now = now_provider or (lambda: datetime.now(timezone.utc).astimezone())
        self.window_origin = window_origin

        if input_driver is not None:
            self.input_driver: InputDriver = input_driver
        elif dry_run:
            self.input_driver = DryRunInputDriver()
        else:
            self.input_driver = PyAutoGuiInputDriver()

    # ------------------------------------------------------------------
    # run_list.json (hot reload sin caché y escritura atómica)
    # ------------------------------------------------------------------

    def load_run_list(self) -> list[str]:
        """Lee ``run_list.json`` desde disco en cada llamada (sin caché)."""
        if not self.run_list_path.is_file():
            logger.warning("run_list.json no existe en %s; devolviendo lista vacía.", self.run_list_path)
            return []

        with self.run_list_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        raw_list = data.get("run_list", [])
        if not isinstance(raw_list, list):
            raise ValueError(f"'run_list' en {self.run_list_path} debe ser una lista.")
        return [str(item).strip() for item in raw_list if str(item).strip()]

    def is_in_run_list(self, name: str) -> bool:
        """Verifica (case-insensitive) si un Loomian figura en ``run_list.json`` en vivo."""
        target = name.strip().casefold()
        if not target:
            return False
        return any(entry.casefold() == target for entry in self.load_run_list())

    def add_to_run_list(self, name: str) -> bool:
        """Agrega un nombre a ``run_list.json`` sin duplicados (case-insensitive) de forma atómica.

        Devuelve ``True`` si el nombre fue agregado, o ``False`` si ya existía o era vacío.
        """
        cleaned = name.strip()
        if not cleaned:
            return False

        self.run_list_path.parent.mkdir(parents=True, exist_ok=True)

        data: dict[str, Any] = {"run_list": []}
        if self.run_list_path.is_file():
            with self.run_list_path.open("r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                data = loaded

        existing = [str(item).strip() for item in data.get("run_list", []) if str(item).strip()]
        if any(entry.casefold() == cleaned.casefold() for entry in existing):
            logger.info("'%s' ya existe en %s; no se duplica.", cleaned, self.run_list_path)
            return False

        existing.append(cleaned)
        data["run_list"] = existing

        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.run_list_path.parent),
            prefix=f".{self.run_list_path.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as tmp_file:
                json.dump(data, tmp_file, indent=2, ensure_ascii=False)
                tmp_file.write("\n")
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
            os.replace(tmp_path, self.run_list_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

        logger.info("Agregado '%s' a %s.", cleaned, self.run_list_path)
        return True

    # ------------------------------------------------------------------
    # data/corruptos.csv
    # ------------------------------------------------------------------

    def append_corrupt(self, loomian: str, shard: str) -> None:
        """Anexa una fila ``timestamp,loomian,shard`` en UTF-8 con timestamp ISO 8601 y flush."""
        self.corruptos_csv_path.parent.mkdir(parents=True, exist_ok=True)
        needs_header = (
            not self.corruptos_csv_path.exists() or self.corruptos_csv_path.stat().st_size == 0
        )
        timestamp = self._now().isoformat(timespec="seconds")

        with self.corruptos_csv_path.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            if needs_header:
                writer.writerow(CSV_HEADER)
            writer.writerow([timestamp, loomian.strip(), shard.strip()])
            fh.flush()

        logger.info("Corrupto registrado en %s: %s (%s)", self.corruptos_csv_path, loomian, shard)

    # ------------------------------------------------------------------
    # Acciones de alto nivel sobre el juego
    # ------------------------------------------------------------------

    def _to_screen_point(self, point: Point) -> Point:
        """Convierte una coordenada relativa a la ventana del juego a coordenada de pantalla."""
        return Point(point.x + self.window_origin.x, point.y + self.window_origin.y)

    def cast_rod(self) -> None:
        """Lanza la caña de pescar."""
        self.input_driver.click(self._to_screen_point(self._require_point("cast_button")))

    def move_to_dot(self, point: Point) -> None:
        """Posiciona el cursor sobre un nodo del minijuego."""
        self.input_driver.move_to(self._to_screen_point(point))

    def click_run(self) -> None:
        """Hace clic en el botón grande de Run en combate."""
        self.input_driver.click(self._to_screen_point(self._require_point("run_button")))

    def click_cancel(self) -> None:
        """Hace clic en el botón grande gris Cancel dentro del submenú de Fight."""
        self.input_driver.click(self._to_screen_point(self._require_point("cancel_button")))

    def click_fight_and_move(self) -> None:
        """Hace clic en Fight y selecciona el movimiento configurado contra corruptos."""
        self.input_driver.click(self._to_screen_point(self._require_point("fight_button")))
        self.input_driver.click(self._to_screen_point(self._require_point("corrupt_move_slot")))

    def close_item_notice(self) -> None:
        """Cierra o avanza el aviso de ítem."""
        point = self.ui_points.get("item_notice_close")
        if point is not None:
            self.input_driver.click(self._to_screen_point(point))

    def _require_point(self, name: str) -> Point:
        if name not in self.ui_points:
            raise KeyError(f"Coordenada UI '{name}' no definida en settings.json.")
        return self.ui_points[name]
