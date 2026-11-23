Automatización de Pesca en Loomian Legacy
1. Objetivo
Automatizar el ciclo repetitivo de pesca en Loomian Legacy:

Lanzar la caña y esperar el minijuego de puntos.

Resolver la secuencia de puntos conectados por líneas (empezando por el punto verde) antes de que se agote la barra superior.

Detectar el resultado del ciclo:

Pesca sin combate (retorno estable a la vista de pesca): esperar confirmación y continuar el ciclo.

Combate contra Loomian: identificar nombre y forma (normal, corrupto, alpha, gamma) y actuar según las reglas definidas.

Garantizar control seguro con kill-switch, modo dry-run/debug y detención con alerta sonora ante casos no automáticos o inesperados.

2. Reglas de Decisión en Combate `IN_BATTLE` (Lógica de Combate y Decisión)

Al ingresar a batalla y detectarse el menú principal (`Fight`, `Items`, `Run`, `Loomians`), el sistema identifica el **nombre** del Loomian y su **forma** (`normal`, `corrupto`, `alpha`, `gamma`) y ejecuta la acción correspondiente:

* **Forma `normal`**:
* Se consulta en vivo `config/run_list.json` (recargada en cada encuentro para reflejar cambios sin reiniciar el bot).
* **Si el Loomian está en `run_list.json`**: se hace clic en el botón grande de **Run**, se espera la salida del combate (ej. *"You got away safely!"* o regreso a la vista de pesca) y se pasa a `COOLDOWN` para continuar el ciclo.
* **Si el Loomian NO está en `run_list.json`**: se dispara la **alerta sonora** (`alerts/sound_alert.py`) y la máquina de estados pasa a `STOPPED_ALERT`, deteniendo la ejecución de forma segura.


* **Forma `corrupto`**:
* Se hace clic en **Fight**, se selecciona el movimiento configurado en `config/settings.json` para golpear/derrotar y obtener el shard.
* Al obtener/cerrar el drop del shard, se anexa una fila en `data/corruptos.csv` con el formato `timestamp,loomian,shard`.
* Según la configuración en `config/settings.json` (`stop_after_corrupt`), el flujo pasa a `COOLDOWN` para seguir pescando o a `STOPPED_ALERT` / pausa segura.


* **Formas `alpha` y `gamma`**:
* Se dispara la **alerta sonora** y se pasa a `STOPPED_ALERT` (o pausa controlada) para no perder el encuentro ni hacer acciones automáticas no deseadas.
* Se habilita una opción rápida (por consola o atajo) para **añadir el nombre del Loomian a `config/run_list.json`** en caso de querer huir automáticamente de esa especie en futuras apariciones.


* **Confianza baja, forma no reconocida o estado ambiguo**:
* Por seguridad, ante cualquier duda en la lectura del nombre, la forma o los botones de batalla, se dispara la **alerta sonora** y se detiene la ejecución (`STOPPED_ALERT`).



3. Estructura del Proyecto
Plaintext
.
├── README.md                    # Guía de instalación, uso (--dry-run/--no-dry-run), kill-switch y calibración
├── requirements.txt             # Dependencias Python permitidas (versiones fijadas)
├── .github/
│   └── workflows/
│       └── tests.yml            # Workflow de CI con pytest
├── assets/
│   └── templates/               # Plantillas PNG para template matching
├── config/
│   ├── settings.json            # ROIs, coordenadas/slots de UI, timeouts, slot/coordenada del movimiento al abrir Fight, rutas
│   ├── run_list.json            # Lista editable en vivo de Loomians normales para hacer RUN
│   └── README.md                # Documentación de claves de settings.json
├── data/
│   └── corruptos.csv            # Cabecera: timestamp,loomian,shard
├── alerts/
│   ├── __init__.py
│   └── sound_alert.py           # Disparo de alerta sonora (WAV / tono del sistema)
├── core/
│   ├── __init__.py
│   ├── types.py                 # Tipos compartidos (State, Form, ROI, Point, EncounterResult)
│   ├── state_machine.py         # Estados, transiciones, timeouts, kill-switch y pausa/stop
│   └── controller.py            # Control de mouse/teclado, lectura/actualización de run_list y CSV
├── vision/
│   ├── __init__.py
│   ├── screen_capture.py        # Captura de pantalla / ROI con rate-limit configurable
│   ├── minigame_tracker.py      # Detección de barra, nodo verde inicial, líneas y orden de nodos
│   └── encounter_classifier.py  # Clasifica retorno/combate, banners, shards y forma del Loomian
├── debug/
│   ├── __init__.py
│   ├── visualizer.py            # Dibuja ROIs, centros, orden de puntos y etiquetas sobre frames/video
│   ├── extract_frames.py        # Extrae frames PNG desde video y recorta ROIs de referencia
│   ├── evaluate_tracker.py      # Evalúa MinigameTracker sobre samples/frames* vs ground_truth.json
│   └── validate_templates.py    # Valida templates de assets/templates/ contra samples/ sin falsos positivos
├── tests/
│   ├── __init__.py
│   ├── test_controller.py
│   ├── test_extract_frames.py
│   ├── test_kill_switch.py
│   ├── test_state_machine.py
│   └── test_vision.py
└── main.py                      # Punto de entrada, flags (--dry-run, --no-dry-run, --debug), logs y loop principal
4. Máquina de Estados
Estados principales con timeout por estado y kill-switch global (F8 o tecla configurada):

IDLE_OR_CAST: lanza la caña y pasa a WAITING_MINIGAME.

WAITING_MINIGAME: busca la barra superior / nodos del minijuego en el ROI correspondiente. Si vence el timeout, reintenta casteo.

SOLVING_DOTS: obtiene la lista ordenada de coordenadas (x, y) siguiendo el nodo verde y las líneas conectadas; mueve/posiciona el cursor sobre cada punto en orden hasta completar o cerrarse el minijuego.

RESOLVING_OUTCOME:

Acumula indicios transitorios (banners de corrupto y destellos alpha/gamma).

Si detecta retorno estable a la vista de pesca (sin oscurecimiento por fade) -> pasa a COOLDOWN.

Si detecta interfaz de combate / menú (Fight, Items, Run, Loomians) -> pasa a IN_BATTLE.

Si vence el timeout sin cambios claros -> pasa a COOLDOWN.

IN_BATTLE: ejecuta la regla de normal (Run / alerta + stop), corrupto (Fight -> movimiento -> log CSV) o alpha/gamma (alerta + stop + opción de agregar a run_list.json).

COOLDOWN: breve pausa de estabilización antes del próximo ciclo.

STOPPED_ALERT / ABORTED: estado final seguro al disparar alerta de detención o activar el kill-switch.

5. Lineamientos Técnicos y de Pruebas
Bajo consumo y desacoplamiento: usar rate-limit en screen_capture.py (menor frecuencia en espera, mayor durante SOLVING_DOTS) y mantener lógica de visión separada del movimiento del mouse.

Modo dry-run y debug: permitir probar minigame_tracker.py y encounter_classifier.py sobre capturas o frames extraídos del video sin ejecutar clics reales.

Iteración limpia: generar primero el esqueleto y los módulos base con interfaces tipadas y docstrings, luego validar visión con visualizer.py y finalmente integrar el loop completo.
