# SkyDock Simulation Engine (8-Inch Quad + Radxa Zero 3W + OV9281)

Entorno de simulación de alta fidelidad para el desarrollo, prueba y validación de algoritmos de aterrizaje de precisión autónomo para el proyecto **SkyDock**.

---

## 🎯 Arquitectura del Sistema

El simulador reproduce de extremo a extremo la cadena de percepción, control y actuación del dron:

1. **Cuadricóptero de 8 Pulgadas (`simulations/config/drone_params.py`)**:
   - Chasis Quad-X de 370 mm con hélices de 8"x4.5" y motores 2806.5.
   - Modelo de dinámica de vuelo 6-DOF con inercias reales, resistencia aerodinámica y gravedad.
2. **Cámara Industrial Nadir OV9281 (`simulations/config/camera_params.py` y `simulations/vision/sim_camera.py`)**:
   - Modelo exacto del módulo **AED7-720P Global Shutter USB Industrial Camera** (OmniVision OV9281).
   - Resolución de 1280x720 (720p) @ 120 FPS monocromático.
   - Renderizador de proyección de perspectiva 3D realista basado en la matriz intrínseca $K$ y la orientación del dron.
3. **Estación de Aterrizaje / Objetivo Visual (`simulations/vision/targets.py`)**:
   - Texturas de alta definición para **Marcador ArUco (DICT_4X4_50)** y **Código QR**.
   - Landing Pad con círculos de aproximación y líneas de alineación de alta visibilidad.
4. **Companion Computer a Bordo - Radxa Zero 3W (`simulations/companion/`)**:
   - **Detector y Estimador de Pose (`target_detector.py`)**: Algoritmo PnP con refinamiento subpixel y Filtro de Kalman 3D para seguimiento continuo incluso durante oclusión momentánea.
   - **Controlador de Aterrizaje (`landing_controller.py`)**: Máquina de estados (`SEARCHING`, `ALIGNING`, `DESCENDING`, `TOUCHDOWN`) con lazos PID desacoplados en los ejes $X, Y, Z$ y control de guiñada.
   - **Puente MAVLink (`mavlink_bridge.py`)**: Comunicación bidireccional PyMavlink transmitiendo `SET_POSITION_TARGET_LOCAL_NED` (control de velocidad en marco del cuerpo) y mensajes `LANDING_TARGET`.
5. **Servidor MAVLink ArduPilot SITL (`simulations/sitl/quad_sitl.py`)**:
   - Servidor MAVLink UDP integrado (puerto `14550`) compatible con ArduPilot, QGroundControl y Mission Planner.
6. **Dashboard Interactivo en Tiempo Real (`simulations/dashboard/sim_dashboard.py` y `simulations/vision/third_person_viewer.py`)**:
   - **Vista Dual Dividida**: Cámara Nadir OV9281 (POV con detección y retícula) + **Vista 3D en Tercera Persona** del cuadricóptero.
   - **Renderizado 3D Completo**: Muestra el dron de 8" con sus 4 brazos de fibra de carbono, motores con colores frontal/trasero, inclinación Roll/Pitch/Yaw en tiempo real, sombra elíptica en el suelo, cono de visión (Frustum) de la cámara hacia el landing pad y estela 3D de trayectoria.
   - Radar 2D de posicionamiento relativo y panel de telemetría / MAVLink.

---

## 🚀 Requisitos e Instalación

Instala las dependencias necesarias:

```bash
pip install -r simulations/requirements.txt
```

---

## 🎮 Ejecución de la Simulación

### 1. Ejecución Básica con Interfaz Gráfica:
```bash
python simulations/run_simulation.py
```

### 2. Parámetros de Simulación Personalizados:
Puedes configurar la altitud inicial, el desplazamiento del dron, el tipo de objetivo visual o simular viento:

```bash
python simulations/run_simulation.py --altitude 4.0 --offset_x 1.5 --offset_y -1.0 --target_type aruco --wind_x 0.8
```

#### Argumentos Disponibles:
- `--altitude <float>`: Altitud inicial del dron en metros (por defecto: `3.0`).
- `--offset_x <float>`: Desplazamiento inicial Norte (metros) (por defecto: `1.2`).
- `--offset_y <float>`: Desplazamiento inicial Este (metros) (por defecto: `-0.8`).
- `--yaw <float>`: Ángulo de guiñada inicial en grados (por defecto: `20.0`).
- `--target_type <aruco|qr>`: Tipo de objetivo visual a renderizar.
- `--wind_x <float>` / `--wind_y <float>`: Componentes de viento constante en m/s.
- `--headless`: Ejecuta en consola sin ventana gráfica para pruebas automatizadas.

---

## ⌨️ Controles Interactivos en Vivo

Durante la simulación con GUI, puedes presionar las siguientes teclas:
- **`[G]`**: **Iniciar / Detener Grabación del Ensayo** (Graba `ensayo_001_...`, `ensayo_002_...` con su `.csv` y `.log`).
- **`[SPACE]`**: Iniciar la secuencia de búsqueda y aterrizaje de precisión autónomo.
- **`[V]`**: **Alternar modos de visualización** (`DUAL`: Cámara Nadir + 3D Tercera Persona | `CHASE_ONLY`: Solo 3D | `POV_ONLY`: Solo Nadir).
- **`[C]`**: **Alternar modo de cámara 3D** (`CHASE CAM`: Siguiendo al dron desde atrás | `ORBIT CAM`: Vista fija observando la plataforma).
- **`[T]`**: Despegar / ascender a 3 metros.
- **`[R]`**: Resetear la posición del dron a las condiciones iniciales.
- **`[L]`**: Forzar modo `LAND` de ArduPilot.
- **`[W / A / S / D]`**: Mover/perturbar manualmente el dron.
- **`[ESC]` o `[Q]`**: Salir y cerrar la simulación.

---

## 📊 Sistema de Ensayos y Logs (`simulations/logs/`)

- **Modo Prueba Libre (Por Defecto)**: Puedes volar, calibrar y probar aterrizajes sin generar archivos basura en disco. El HUD muestra `[○ MODO LIBRE (NO GRABANDO)]`.
- **Grabación Bajo Demanda (`[G]`)**: Al presionar `[G]`, el sistema inicia un nuevo ensayo numerado consecutivamente:
  - `ensayo_001_YYYYMMDD_HHMMSS.log`: Eventos del sistema, comandos MAVLink y cambios de modo con timestamp milimétrico.
  - `ensayo_001_YYYYMMDD_HHMMSS.csv`: Serie temporal de telemetría a 50-60 Hz (posición $X,Y,Z$, velocidades comandadas $V_x,V_y,V_z$, altitud AGL, etc.).
- **Finalización de Ensayo**: Al presionar `[G]` nuevamente o al tocar tierra y completar la misión, el ensayo se cierra y queda guardado automáticamente.
- Posición 3D estimada del objetivo en el marco del cuerpo $(x, y, z)$.
- Comandos de velocidad MAVLink transmitidos $(V_x, V_y, V_z)$.
- Altitud real del dron sobre el terreno.
