Actúa como un ingeniero de software senior en Python.

# CONTEXTO

- Lee `SPEC.md` (fuente de verdad). Si algo de este prompt contradice SPEC.md, o si hace falta algo que SPEC.md no contempla (por ejemplo `assets/templates/`, `tests/`, `requirements.txt` o `core/types.py`), consúltame antes de decidir o, si el cambio es menor y no toca la lógica, adáptalo y actualiza SPEC.md para que ambos queden alineados.
- Nunca cambies por tu cuenta las reglas `IN_BATTLE`, los estados, las transiciones ni los nombres de módulos definidos en SPEC.md: ante cualquier duda en esos puntos, consúltame siempre.
- Toda adaptación o actualización de SPEC.md debe quedar registrada en la ENTREGA FINAL.
- Entorno: Windows, Python 3.11+, el juego (Loomian Legacy en Roblox) corre en ventana con tamaño fijo.
- Dependencias permitidas: `numpy`, `opencv-python`, `mss`, `pyautogui` (o `pydirectinput`), `keyboard`, `pytest`. No agregues ninguna otra sin consultarme. Genera `requirements.txt`.
- Esta iteración NO debe ejecutar clics ni teclas reales por defecto. `--dry-run` es el valor por defecto; para modo real debe pasarse `--no-dry-run`.
- Código modular, legible y de bajo consumo de recursos. Type hints y docstrings en todos los módulos.

# PASO 0 — PLAN (antes de escribir código)

Preséntame un plan con:
1. Árbol de archivos final.
2. Tipos compartidos.
3. Tabla de transiciones de la máquina de estados.
4. Qué módulos son funcionales y cuáles son stubs.

Espera mi aprobación antes de continuar.

# PASO 1 — ESTRUCTURA

Crea exactamente la estructura de SPEC.md:

- `config/settings.json`, `config/run_list.json`
- `data/corruptos.csv`
- `alerts/sound_alert.py`
- `core/state_machine.py`, `core/controller.py`
- `vision/screen_capture.py`, `vision/minigame_tracker.py`, `vision/encounter_classifier.py`
- `debug/visualizer.py`
- `main.py`

Además: `requirements.txt`, `tests/`, un `__init__.py` por paquete y `assets/templates/` (única carpeta fuera del SPEC, solo para imágenes de template matching).

# PASO 2 — TIPOS COMPARTIDOS

Define (por ejemplo en `core/types.py`):
- `State` (Enum) con los estados del SPEC.
- `Form` (Enum): normal, corrupto, alpha, gamma, unknown.
- `ROI` y `Point` (dataclasses).
- `EncounterResult` (dataclass): nombre, forma, confianza, ítem detectado.

# PASO 3 — CONFIGURACIÓN

**`config/settings.json`**: JSON válido, SIN comentarios. Documenta cada clave con un campo hermano `_doc` o en `config/README.md`. Incluye:
- ROIs.
- Coordenadas/slots de UI: botón Run, Fight, slot de movimiento, aviso de ítem.
- Timeouts por estado.
- Rate-limit de captura (en espera vs. SOLVING_DOTS).
- `kill_switch_key` = "F8".
- `stop_after_corrupt` (booleano).
- `window_size` (ancho y alto esperados del juego).
- `template_match_threshold`, `min_confidence` y `templates_dir`.
- Rutas de archivos.

**`config/run_list.json`**: `{"run_list": ["Goppie", "Cavenish"]}`.

# PASO 4 — `core/controller.py` (funcional)

- Lee `run_list.json` en cada encuentro (sin caché).
- `add_to_run_list(name)`: comparación case-insensitive, sin duplicados, escritura atómica (archivo temporal + replace), `indent=2`, sin perder formato.
- `append_corrupt(loomian, shard)`: crea la cabecera `timestamp,loomian,shard` si falta, UTF-8, `newline=""`, timestamp ISO 8601, flush al escribir.
- Acciones de mouse/teclado detrás de una interfaz; en dry-run solo se loguean.

# PASO 5 — MÓDULOS FUNCIONALES

**`core/state_machine.py`**
- Todos los estados y las reglas `IN_BATTLE` del SPEC.
- Reloj y dependencias (captura, classifier, controller) inyectados para poder testear con fakes.
- Timeout por estado y kill-switch.
- En `alpha`/`gamma`: dispara alerta, pasa a `STOPPED_ALERT` y recién ahí ofrece (por consola) agregar el nombre a `run_list.json`. No abrir hilos ni leer `input()` mientras el loop corre.

**`main.py`**
- Flags `--dry-run` (por defecto), `--no-dry-run` y `--debug`; logging.
- Kill-switch global con `keyboard` (tecla F8, configurable), encapsulado en un adaptador `KillSwitch` con interfaz propia para reemplazarlo por un fake en tests.
- Si el hook no se puede registrar (permisos, etc.):
  - En dry-run: warning claro en el log y continuar.
  - En modo real: abortar ANTES de arrancar, indicando que puede requerir ejecutar la terminal como administrador.

**`alerts/sound_alert.py`**: alerta sonora con fallback si no hay WAV.

**`debug/visualizer.py`**: dibuja ROIs, centros, orden de puntos y etiquetas sobre frames.

# PASO 6 — MÓDULOS DE VISIÓN (stubs)

**`vision/encounter_classifier.py`**
- Nombre del Loomian y elementos de UI (Fight, Items, Run, Loomians, aviso de ítem): diseño basado en template matching con OpenCV. Sin OCR en esta iteración.
- Forma: método separado `classify_form(frame) -> tuple[Form, float]`, solo stub con TODOs claros; se definirá con capturas reales.
- Los templates viven en `assets/templates/`; la ruta se toma de settings.json.
- Si el tamaño del ROI no coincide con `window_size`, o la confianza está bajo el umbral, devolver `Form.UNKNOWN` / nombre desconocido (el state machine lo manda a `STOPPED_ALERT`, según el SPEC).

**`vision/minigame_tracker.py`**: stub con firmas claras y TODOs (barra superior, nodo verde inicial, líneas, orden de nodos).

**`vision/screen_capture.py`**: captura de pantalla/ROI con `mss` y rate-limit configurable.

# PASO 7 — TESTS Y VERIFICACIÓN

Tests con `pytest` para:
- Reglas `IN_BATTLE` (con un classifier falso): normal en run_list, normal fuera de run_list, corrupto, alpha, gamma y confianza baja.
- `run_list`: hot reload y agregar nombres.
- CSV: creación de cabecera y append.
- Timeouts por estado.
- Kill-switch (con el adaptador falso).

Ejecuta `pytest` y `python main.py --dry-run --debug` y muéstrame el resultado.

# ENTREGA FINAL

Resumen breve con:
- Qué quedó funcional.
- Qué quedó como TODO.
- Decisiones que tomaste por tu cuenta.
- Diferencias respecto de SPEC.md y cómo las resolviste (consultadas o adaptadas, y qué cambiaste en SPEC.md).
- Dudas abiertas.
