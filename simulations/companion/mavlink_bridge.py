"""
Puente de Comunicación MAVLink (PyMavlink) para la Radxa Zero 3W.
Maneja la conexión con el piloto automático ArduPilot (SITL o vehículo real),
envía heartbeats, comandos de modo de vuelo y paquetes SET_POSITION_TARGET_LOCAL_NED / LANDING_TARGET.
"""
import time
import threading
import numpy as np
from typing import Optional, Dict, Any, Tuple
from pymavlink import mavutil
from simulations.config.logger import log_event

class MavlinkBridge:
    def __init__(self, connection_str: str = "udpout:127.0.0.1:14550",
                 source_system: int = 1, source_component: int = 191):
        """
        :param connection_str: Cadena de conexión PyMavlink (ej. 'udpout:127.0.0.1:14550' o '/dev/ttyAML0:921600')
        :param source_system: System ID de la Radxa Zero 3W
        :param source_component: Component ID (191 = MAV_COMP_ID_ONBOARD_COMPUTER)
        """
        self.connection_str = connection_str
        self.source_system = source_system
        self.source_component = source_component
        self.master = None
        self.is_connected = False
        self.running = False
        
        # Telemetría recibida
        self.telemetry = {
            "armed": False,
            "flight_mode": "UNKNOWN",
            "pos_ned": np.array([0.0, 0.0, 0.0]), # [x_north, y_east, z_down]
            "vel_ned": np.array([0.0, 0.0, 0.0]),
            "attitude_rad": np.array([0.0, 0.0, 0.0]), # [roll, pitch, yaw]
            "altitude_m": 0.0,
            "battery_pct": 100.0,
            "last_heartbeat_time": 0.0
        }
        self.telemetry_lock = threading.Lock()
        self._rx_thread = None
        self._hb_thread = None

    def connect(self, timeout_s: float = 10.0) -> bool:
        """Establece conexión MAVLink con el piloto automático."""
        log_event(f"[MAVLINK] Conectando a {self.connection_str}...")
        try:
            self.master = mavutil.mavlink_connection(
                self.connection_str,
                source_system=self.source_system,
                source_component=self.source_component
            )
            
            # Enviar un primer heartbeat para que el servidor SITL registre la IP/puerto del cliente
            self.master.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                0, 0, 0
            )

            # Esperar primer heartbeat
            msg = self.master.wait_heartbeat(timeout=timeout_s)
            if msg is not None:
                self.is_connected = True
                self.running = True
                log_event(f"[MAVLINK] Conexión establecida con Sistema {self.master.target_system}, Componente {self.master.target_component}")
                
                # Iniciar hilos de recepción y heartbeat
                self._rx_thread = threading.Thread(target=self._telemetry_receiver_loop, daemon=True)
                self._rx_thread.start()
                
                self._hb_thread = threading.Thread(target=self._heartbeat_sender_loop, daemon=True)
                self._hb_thread.start()
                return True
            else:
                log_event("[MAVLINK] Timeout esperando heartbeat de ArduPilot.")
                return False
        except Exception as e:
            log_event(f"[MAVLINK] Error al conectar: {e}")
            return False

    def _heartbeat_sender_loop(self):
        """Envía latidos de corazón (HEARTBEAT) a 1 Hz para mantener viva la conexión."""
        while self.running:
            try:
                if self.master:
                    self.master.mav.heartbeat_send(
                        mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                        mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                        0, 0, 0
                    )
            except Exception as e:
                pass
            time.sleep(1.0)

    def _telemetry_receiver_loop(self):
        """Hilo de escucha continua de mensajes MAVLink."""
        while self.running:
            try:
                msg = self.master.recv_match(blocking=True, timeout=0.1)
                if not msg:
                    continue
                
                msg_type = msg.get_type()
                now = time.time()

                with self.telemetry_lock:
                    if msg_type == 'HEARTBEAT':
                        self.telemetry["last_heartbeat_time"] = now
                        self.telemetry["armed"] = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                        # Decodificar modo de vuelo ArduPilot
                        custom_mode = msg.custom_mode
                        mode_map = self.master.mode_mapping()
                        if mode_map:
                            for name, mode_id in mode_map.items():
                                if mode_id == custom_mode:
                                    self.telemetry["flight_mode"] = name
                                    break

                    elif msg_type == 'LOCAL_POSITION_NED':
                        self.telemetry["pos_ned"] = np.array([msg.x, msg.y, msg.z], dtype=np.float64)
                        self.telemetry["vel_ned"] = np.array([msg.vx, msg.vy, msg.vz], dtype=np.float64)
                        self.telemetry["altitude_m"] = -msg.z

                    elif msg_type == 'ATTITUDE':
                        self.telemetry["attitude_rad"] = np.array([msg.roll, msg.pitch, msg.yaw], dtype=np.float64)

                    elif msg_type == 'SYS_STATUS':
                        self.telemetry["battery_pct"] = float(msg.battery_remaining)

            except Exception as e:
                time.sleep(0.01)

    def get_telemetry(self) -> Dict[str, Any]:
        """Obtiene una copia segura de los datos de telemetría."""
        with self.telemetry_lock:
            return self.telemetry.copy()

    def set_mode(self, mode_name: str):
        """Cambia el modo de vuelo en ArduPilot (ej. 'GUIDED', 'LAND', 'STABILIZE')."""
        if not self.master:
            return
        log_event(f"[MAVLINK] Solicitando cambio a modo {mode_name}...")
        mode_id = self.master.mode_mapping().get(mode_name.upper()) if self.master.mode_mapping() else None
        if mode_id is not None:
            self.master.set_mode(mode_id)
        else:
            # Fallback manual para ArduCopter
            modes = {"STABILIZE": 0, "ALT_HOLD": 2, "AUTO": 3, "GUIDED": 4, "LOITER": 5, "RTL": 6, "LAND": 9}
            mid = modes.get(mode_name.upper(), 4)
            self.master.mav.set_mode_send(
                self.master.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mid
            )

    def arm_disarm(self, arm: bool):
        """Arma (True) o desarma (False) los motores."""
        if not self.master:
            return
        param1 = 1.0 if arm else 0.0
        log_event(f"[MAVLINK] {'ARMANDO' if arm else 'DESARMANDO'} motores...")
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            param1, 0, 0, 0, 0, 0, 0
        )

    def send_velocity_body_ned(self, vx: float, vy: float, vz: float, yaw_rate_rad_s: float = 0.0):
        """
        Envía vector de velocidad en el marco del cuerpo (Forward, Right, Down) a ArduPilot.
        
        :param vx: Velocidad hacia adelante (m/s)
        :param vy: Velocidad hacia la derecha (m/s)
        :param vz: Velocidad hacia abajo (m/s) (positivo = descender)
        :param yaw_rate_rad_s: Velocidad de giro en guiñada (rad/s)
        """
        if not self.master:
            return
        
        # Máscara de bits: ignorar posición y aceleración, usar solo velocidad (vx, vy, vz) y yaw_rate
        # Bitmask: 0b0000_1001_1100_0111 = 2503 (decimal)
        type_mask = (
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
        )

        self.master.mav.set_position_target_local_ned_send(
            int(time.time() * 1000) & 0xFFFFFFFF, # time_boot_ms
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED,   # Marco local del cuerpo FRD
            type_mask,
            0, 0, 0,                             # Posición (ignorada)
            float(vx), float(vy), float(vz),     # Velocidades (m/s)
            0, 0, 0,                             # Aceleraciones (ignoradas)
            0.0,                                 # Yaw (ignorado)
            float(yaw_rate_rad_s)                # Yaw rate (rad/s)
        )

    def send_landing_target(self, angle_x_rad: float, angle_y_rad: float, distance_m: float):
        """Envía mensaje estándar MAVLink LANDING_TARGET para ArduPilot precision landing."""
        if not self.master:
            return
        self.master.mav.landing_target_send(
            int(time.time() * 1e6) & 0xFFFFFFFFFFFFFFFF, # time_usec
            0, # target num
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            float(angle_x_rad),
            float(angle_y_rad),
            float(distance_m),
            0.20, 0.20 # size_x, size_y
        )

    def close(self):
        """Cierra hilos y conexión."""
        self.running = False
        if self.master:
            try:
                self.master.close()
            except Exception:
                pass
            self.master = None
        self.is_connected = False
