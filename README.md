# Loomian Legacy Fishing Bot

Bot modular en Python para automatizar el ciclo de pesca en **Loomian Legacy** (Roblox en Windows) con máquina de estados determinista, modo `dry-run` seguro por defecto, kill-switch global y depuración visual sobre capturas reales.

> Fuente de verdad de estados y reglas de combate: [SPEC.md](SPEC.md).

## Qué hace

1. **Lanza la caña (`IDLE_OR_CAST` → `WAITING_MINIGAME`)**: espera la aparición del minijuego de puntos.
2. **Resuelve el minijuego (`SOLVING_DOTS`)**: detecta la barra superior de tiempo y recorre la secuencia conectada desde el nodo verde inicial.
3. **Evalúa el resultado (`RESOLVING_OUTCOME`)**:
   - **Ítem basura**: cierra el aviso y vuelve a pescar (`COOLDOWN`).
   - **Combate (`IN_BATTLE`)**:
     - `normal` en `config/run_list.json` (recargado en vivo): clic en **Run** y confirma la salida del combate antes de pasar a `COOLDOWN`.
     - `normal` fuera de `config/run_list.json`: alerta sonora y detención segura (`STOPPED_ALERT`).
     - `corrupto`: clic en **Fight** + movimiento configurado, espera el drop del shard y el fin del combate, registra `timestamp,loomian,shard` en `data/corruptos.csv` y continúa o se detiene según `stop_after_corrupt`.
     - `alpha` / `gamma` o baja confianza / estado ambiguo: alerta sonora y `STOPPED_ALERT` (ofreciendo en consola al detenerse agregar la especie a `run_list.json`).

## Requisitos e instalación

- **Sistema operativo**: Windows
- **Python**: 3.11+
- **Roblox**: ventana de tamaño fijo calibrada en `config/settings.json` (`1680x976` por defecto).

```powershell
python -m pip install -r requirements.txt
```

## Cómo ejecutar

### Modo seguro por defecto (`--dry-run`)
No ejecuta clics ni teclas reales (solo registra las acciones en el log):

```powershell
python main.py --dry-run --debug
```

Para pruebas acotadas:

```powershell
python main.py --dry-run --debug --max-steps 5
```

### Modo real (`--no-dry-run`)
Ejecuta clics y movimientos reales con `pyautogui`:

```powershell
python main.py --no-dry-run
```

## Kill-switch global (`F8`)

- La tecla de emergencia configurable en `config/settings.json` (`kill_switch_key`, por defecto `"F8"`) detiene el bot inmediatamente pasando a `ABORTED`.
- En **Windows**, el hook global de `keyboard` puede requerir **ejecutar la terminal como Administrador**:
  - En `--dry-run`, si el hook no se puede registrar, se emite un warning y continúa.
  - En `--no-dry-run`, el programa **aborta antes de arrancar** si no logra registrar el hook.

## Calibración de `config/settings.json` y herramientas de debug

1. **Extraer frames y recortar plantillas PNG** desde un video local:
   ```powershell
   python -m debug.extract_frames extract samples\clip1_minijuego.mp4 -o samples\frames -i 1.0 --expected-size 1680 976
   python -m debug.extract_frames crop samples\frames\frame_0004_000s.png --roi 1410 200 230 70 -o assets\templates\loomians\Goppie.png --preview samples\previews\goppie_preview.png
   ```
2. **Evaluar el tracker del minijuego** sobre los frames extraídos:
   ```powershell
   python -m debug.evaluate_tracker
   ```
3. Ajustar `window_size`, `window_origin`, `rois`, `ui_points` y `solving` en `config/settings.json` siguiendo [config/README.md](config/README.md).

## Pruebas

```powershell
python -m pytest -v
```
