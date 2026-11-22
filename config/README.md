# Documentación de `config/settings.json`

Todas las coordenadas y tamaños están expresados en píxeles relativos a la ventana del juego con tamaño fijo (`window_size`).

## Claves principales

- **`window_size`**: Dimensiones esperadas (`width`, `height`) del área cliente de la ventana de Loomian Legacy en Roblox (ej. `1680x976`). Si el frame recibido por el clasificador no coincide con estas dimensiones, se rechaza la lectura por seguridad (`Form.UNKNOWN`).
- **`window_origin`**: Coordenada superior izquierda (`x`, `y`) del área cliente de Roblox en el monitor (ej. `x=0, y=32` para descontar la barra de título en un monitor `1680x1050`). Se usa para capturar exactamente `window_size` en `vision/screen_capture.py` y trasladar clics/movimientos de ventana a coordenadas de pantalla en `core/controller.py`.
- **`rois`**: Regiones de interés (`x`, `y`, `width`, `height`) para captura y visión:
  - `minigame`: Área donde aparecen la barra superior y los puntos conectados del minijuego de pesca.
  - `battle_menu`: Área del menú principal de combate (`Fight`, `Items`, `Run`, `Loomians`).
  - `loomian_name`: Área donde aparece el nombre del Loomian salvaje.
  - `loomian_sprite`: Área del modelo/efectos visuales del Loomian para detectar su forma.
  - `item_notice`: Área del aviso emergente de ítem basura o drop de shard.
- **`ui_points`**: Coordenadas (`x`, `y`) de los botones y slots interactivos:
  - `cast_button`: Posición de clic para lanzar la caña.
  - `fight_button`: Botón `Fight` en combate.
  - `run_button`: Botón grande `Run` en combate.
  - `cancel_button`: Botón grande gris `Cancel` dentro del submenú de `Fight` para volver al menú principal de combate.
  - `corrupt_move_slot`: Coordenada del slot de movimiento a usar contra un Loomian corrupto tras abrir `Fight`.
  - `item_notice_close`: Posición para cerrar/avanzar el aviso de ítem si requiere clic.
- **`timeouts`**: Tiempo máximo en segundos por cada estado de la máquina de estados (`IDLE_OR_CAST`, `WAITING_MINIGAME`, `SOLVING_DOTS`, `RESOLVING_OUTCOME`, `IN_BATTLE`, `COOLDOWN`).
- **`capture_rate_limit`**: Control de FPS máximo de captura en `vision/screen_capture.py`:
  - `idle_fps`: Frecuencia baja en espera y estados no críticos (bajo consumo).
  - `solving_dots_fps`: Frecuencia alta durante `SOLVING_DOTS`.
- **`kill_switch_key`**: Tecla global de detención de emergencia (por defecto `"F8"`).
- **`stop_after_corrupt`**: Booleano. Si es `true`, tras derrotar y registrar un corrupto pasa a `STOPPED_ALERT`; si es `false`, pasa a `COOLDOWN` y continúa pescando.
- **`template_match_threshold`**: Umbral mínimo (`0.0` a `1.0`) para aceptar una coincidencia de template matching con OpenCV.
- **`min_confidence`**: Confianza mínima (`0.0` a `1.0`) exigida en `IN_BATTLE` para ejecutar acciones automáticas. Debajo de este umbral se dispara `STOPPED_ALERT`.
- **`templates_dir`**: Carpeta con las plantillas PNG de UI y nombres de Loomians (`assets/templates`).
- **`paths`**: Rutas de archivos de datos y recursos:
  - `run_list`: Ruta a `config/run_list.json` (recargado en vivo en cada encuentro).
  - `corruptos_csv`: Ruta a `data/corruptos.csv`.
  - `alert_wav`: Ruta opcional a un archivo `.wav` para la alerta sonora (si no existe, se usa el tono del sistema como fallback).
