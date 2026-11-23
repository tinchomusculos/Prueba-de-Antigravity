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
  - `battle_banner`: Área del banner azul inferior (banners de encuentro corrupto, `"You got away safely!"` y drop de shard en 1 o 2 líneas).
  - `enemy_card`: Área de la tarjeta superior derecha del Loomian enemigo (celeste en normal, violeta en corrupto).
- **`ui_points`**: Coordenadas (`x`, `y`) de los botones y slots interactivos:
  - `cast_button`: Posición de clic para lanzar la caña.
  - `fight_button`: Botón `Fight` en combate.
  - `run_button`: Botón grande `Run` en combate.
  - `cancel_button`: Botón grande gris `Cancel` dentro del submenú de `Fight` para volver al menú principal de combate. Comparte intencionalmente las coordenadas `(1095, 905)` con `run_button` porque en Loomian Legacy el hexágono inferior central cambia de `Run` (verde) a `Cancel` (gris) en la misma posición al abrir `Fight`.
  - `corrupt_move_slot`: Coordenada del slot de movimiento a usar contra un Loomian corrupto tras abrir `Fight`.
  - `item_notice_close`: Posición para cerrar/avanzar el banner de drop de shard tras derrotar un corrupto.
- **`timeouts`**: Tiempo máximo en segundos por cada estado de la máquina de estados (`IDLE_OR_CAST`, `WAITING_MINIGAME`, `SOLVING_DOTS`, `RESOLVING_OUTCOME`, `IN_BATTLE`, `COOLDOWN`).
- **`capture_rate_limit`**: Control de FPS máximo de captura en `vision/screen_capture.py`:
  - `idle_fps`: Frecuencia baja en espera y estados no críticos (bajo consumo).
  - `resolving_outcome_fps`: Frecuencia intermedia en `RESOLVING_OUTCOME` para capturar banners transitorios y destellos antes de entrar a `IN_BATTLE`.
  - `solving_dots_fps`: Frecuencia alta durante `SOLVING_DOTS`.
- **`resolving`**: Parámetros de transición `RESOLVING_OUTCOME -> COOLDOWN`:
  - `fishing_return_confirm_s`: Tiempo continuo en segundos que debe mantenerse visible la vista de pesca antes de aceptar `COOLDOWN` (calibrado inicialmente solo con `clip3`; validar con más clips).
  - `dark_frame_mean_threshold`: Brillo medio mínimo del frame (`0..255`) para aceptar retorno a pesca; si el frame se oscurece (fade a combate), no pasa a `COOLDOWN`.
- **`form_detection`**: Parámetros de clasificación de forma y shards:
  - `alpha_sparkle_hsv_lower` / `alpha_sparkle_hsv_upper`: Rango HSV hipótesis para destellos celestes (`alpha`), configurable hasta calibrar con clips reales.
  - `gamma_sparkle_hsv_lower` / `gamma_sparkle_hsv_upper`: Rango HSV hipótesis para destellos verdes (`gamma`), configurable hasta calibrar con clips reales.
  - `sparkle_min_pixels`: Cantidad mínima de píxeles brillantes para considerar un destello.
  - `normal_palette_min_score`: Similitud mínima (`0.0` a `1.0`) del histograma de color del sprite contra la paleta normal de la especie.
  - `form_confirm_frames`: Ventana mínima de frames sin destellos/wisp para confirmar forma normal.
  - `unknown_shards_dir`: Carpeta donde se guardan capturas de drops confirmados cuyo tipo de shard no pudo leerse.
- **`solving`**: Parámetros de ejecución dentro de `SOLVING_DOTS`:
  - `dot_move_delay_s`: Pausa en segundos entre movimientos sucesivos de puntos (`0.04` por defecto).
  - `min_bar_ratio`: Proporción mínima restante en la barra superior (`0.0` a `1.0`) para seguir intentando mover entre nodos antes de pasar a `RESOLVING_OUTCOME`.
  - `max_solving_retries`: Máximo de capturas consecutivas sin nodos ordenados antes de pasar a `RESOLVING_OUTCOME`.
- **`kill_switch_key`**: Tecla global de detención de emergencia (por defecto `"F8"`).
- **`stop_after_corrupt`**: Booleano. Si es `true`, tras derrotar y registrar un corrupto pasa a `STOPPED_ALERT`; si es `false`, pasa a `COOLDOWN` y continúa pescando.
- **`template_match_threshold`**: Umbral mínimo (`0.0` a `1.0`) para aceptar una coincidencia de template matching con OpenCV.
- **`min_confidence`**: Confianza mínima (`0.0` a `1.0`) exigida en `IN_BATTLE` para ejecutar acciones automáticas. Debajo de este umbral se dispara `STOPPED_ALERT`.
- **`templates_dir`**: Carpeta con las plantillas PNG de UI y nombres de Loomians (`assets/templates`).
- **`paths`**: Rutas de archivos de datos y recursos:
  - `run_list`: Ruta a `config/run_list.json` (recargado en vivo en cada encuentro).
  - `corruptos_csv`: Ruta a `data/corruptos.csv`.
  - `alert_wav`: Ruta opcional a un archivo `.wav` para la alerta sonora (si no existe, se usa el tono del sistema como fallback).
