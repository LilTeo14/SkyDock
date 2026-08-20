"""
Nodo Principal de la Computadora a Bordo (Radxa Zero 3W).
Integra adquisición de video OV9281, detección de visión, estimación de pose,
controlador de aterrizaje de precisión y envío de comandos MAVLink al piloto automático ArduPilot.
"""
import time
import csv
import threading
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, Callable

from simulations.config.camera_params import CAMERA_PARAMS
from simulations.companion.target_detector import VisualTargetDetector
from simulations.companion.landing_controller import AutonomousPrecisionLandingController, LandingState
from simulations.companion.mavlink_bridge import MavlinkBridge

from simulations.config.logger import SkyDockLogger, log_event

class RadxaCompanionNode:
    def __init__(self, mavlink_conn_str: str = "udpout:127.0.0.1:14550",
                 enable_csv_logging: bool = False,
                 target_loop_rate_hz: float = 60.0):
        self.mavlink_conn_str = mavlink_conn_str
        self.enable_csv_logging = enable_csv_logging
        self.target_loop_rate_hz = target_loop_rate_hz
        
        # Módulos internos
        self.detector = VisualTargetDetector(camera_params=CAMERA_PARAMS)
        self.controller = AutonomousPrecisionLandingController()
        self.mavlink = MavlinkBridge(connection_str=mavlink_conn_str)
        self.logger_manager = SkyDockLogger.get_logger()
        
        # Estado de ejecución
        self.running = False
        self.loop_thread = None
        self.frame_provider: Optional[Callable[[], np.ndarray]] = None
        
        # Métricas en tiempo real
        self.last_detection_data: Dict[str, Any] = {}
        self.last_cmd_vel: np.ndarray = np.zeros(3)
        self.last_state: LandingState = LandingState.IDLE
        self.fps_actual = 0.0
        
        # Logger CSV de Ensayo
        self.csv_file = None
        self.csv_writer = None
        self.is_recording_trial = False
        self.current_trial_num = 0

    def set_frame_provider(self, provider: Callable[[], np.ndarray]):
        """Asigna la fuente de fotogramas (cámara sintética o captura de hardware)."""
        self.frame_provider = provider

    def start(self, auto_connect_mavlink: bool = True):
        """Inicia el ciclo de ejecución de la Radxa Zero 3W."""
        if self.running:
            return
        
        if auto_connect_mavlink:
            self.mavlink.connect(timeout_s=5.0)

        if self.enable_csv_logging:
            self.start_trial_recording()

        self.running = True
        self.loop_thread = threading.Thread(target=self._main_control_loop, daemon=True)
        self.loop_thread.start()
        log_event(f"[RADXA ZERO 3W] Nodo de control activo ({self.target_loop_rate_hz} Hz). Modo libre listo.")

    def start_trial_recording(self) -> Tuple[int, str]:
        """Inicia la grabación de un ensayo formal numerado."""
        if self.is_recording_trial:
            self.stop_trial_recording()

        trial_num, trial_id = self.logger_manager.start_trial()
        self.current_trial_num = trial_num
        
        # Crear archivo CSV para este ensayo
        csv_path = self.logger_manager.logs_dir / f"{trial_id}.csv"
        self.csv_file = open(str(csv_path), mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "timestamp", "flight_mode", "state", "detected", "kalman_active",
            "pos_body_x", "pos_body_y", "pos_body_z",
            "vel_cmd_x", "vel_cmd_y", "vel_cmd_z",
            "drone_alt_m", "loop_dt_ms"
        ])
        self.is_recording_trial = True
        log_event(f"[RADXA ZERO 3W] Telemetría CSV guardando en: {csv_path.name}")
        return trial_num, trial_id

    def stop_trial_recording(self) -> Optional[int]:
        """Detiene y cierra la grabación del ensayo actual."""
        if not self.is_recording_trial:
            return None

        trial_num = self.current_trial_num
        if self.csv_file:
            try:
                self.csv_file.close()
            except Exception:
                pass
            self.csv_file = None
            self.csv_writer = None

        self.is_recording_trial = False
        self.logger_manager.stop_trial()
        return trial_num

    def toggle_trial_recording(self) -> bool:
        """Alterna entre grabar ensayo y modo prueba libre."""
        if self.is_recording_trial:
            self.stop_trial_recording()
            return False
        else:
            self.start_trial_recording()
            return True

    def _main_control_loop(self):
        dt_target = 1.0 / self.target_loop_rate_hz
        last_time = time.time()
        frame_counter = 0
        fps_timer = time.time()

        while self.running:
            t_start = time.time()

            # 1. Obtener frame actual de la cámara
            frame = None
            if self.frame_provider is not None:
                frame = self.frame_provider()

            # 2. Obtener telemetría del piloto automático
            telem = self.mavlink.get_telemetry()
            current_altitude = telem.get("altitude_m", 0.0)
            drone_attitude = telem.get("attitude_rad", np.zeros(3))
            flight_mode = telem.get("flight_mode", "UNKNOWN")

            # 3. Procesar visión por computadora si hay fotograma
            if frame is not None:
                self.last_detection_data = self.detector.detect(frame, current_attitude_rad=drone_attitude)
            else:
                self.last_detection_data = {"detected": False, "kalman_active": False}

            # 4. Actualizar controlador de aterrizaje autónomo
            vel_cmd, yaw_rate_cmd, state = self.controller.update(
                self.last_detection_data, current_altitude=current_altitude
            )
            self.last_cmd_vel = vel_cmd
            self.last_state = state

            # 5. Enviar comandos por MAVLink si estamos en modo autónomo
            if self.mavlink.is_connected:
                # Si estamos en modo de descenso, alineación, búsqueda activa o takeoff, enviar velocidades
                if state in [LandingState.TAKEOFF, LandingState.SEARCHING, LandingState.ALIGNING, LandingState.DESCENDING, LandingState.TOUCHDOWN]:
                    self.mavlink.send_velocity_body_ned(
                        vx=vel_cmd[0], vy=vel_cmd[1], vz=vel_cmd[2],
                        yaw_rate_rad_s=yaw_rate_cmd
                    )
                    
                    # Si alcanzamos contacto final, solicitar cambio a LAND una sola vez
                    if state == LandingState.TOUCHDOWN and current_altitude <= 0.12:
                        if flight_mode != "LAND":
                            self.mavlink.set_mode("LAND")

            # 6. Registrar datos en CSV
            if self.csv_writer is not None:
                pos_body = self.last_detection_data.get("pos_body_filtered")
                pos_x = f"{pos_body[0]:.4f}" if pos_body is not None else ""
                pos_y = f"{pos_body[1]:.4f}" if pos_body is not None else ""
                pos_z = f"{pos_body[2]:.4f}" if pos_body is not None else ""
                
                self.csv_writer.writerow([
                    time.time(), flight_mode, state.value,
                    self.last_detection_data.get("detected", False),
                    self.last_detection_data.get("kalman_active", False),
                    pos_x, pos_y, pos_z,
                    f"{vel_cmd[0]:.3f}", f"{vel_cmd[1]:.3f}", f"{vel_cmd[2]:.3f}",
                    f"{current_altitude:.3f}", f"{(time.time() - t_start)*1000:.2f}"
                ])

            # 7. Contador de FPS y sleep de tasa
            frame_counter += 1
            if time.time() - fps_timer >= 1.0:
                self.fps_actual = frame_counter / (time.time() - fps_timer)
                frame_counter = 0
                fps_timer = time.time()

            elapsed = time.time() - t_start
            sleep_time = max(0.0, dt_target - elapsed)
            time.sleep(sleep_time)

    def trigger_landing_sequence(self):
        """Inicia manualmente la secuencia de búsqueda y aterrizaje de precisión."""
        log_event("[RADXA ZERO 3W] Activando secuencia de aterrizaje autónomo...")
        if self.mavlink.is_connected:
            self.mavlink.set_mode("GUIDED")
        self.controller.set_state(LandingState.SEARCHING)

    def stop(self):
        """Detiene el nodo de la Radxa y cierra recursos."""
        self.running = False
        if self.loop_thread:
            self.loop_thread.join(timeout=1.0)
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
        self.mavlink.close()
        log_event("[RADXA ZERO 3W] Nodo detenido.")
