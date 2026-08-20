"""
Simulador Físico 6-DOF para Cuadricóptero de 8 Pulgadas con Servidor MAVLink ArduPilot SITL Integrado.
Permite ejecutar simulaciones realistas de vuelo, dinámica de viento, respuesta de motores
y responder a comandos MAVLink estándar sin necesidad de software externo complejo.
"""
import time
import socket
import threading
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
from typing import Dict, Any, Tuple, Optional
from pymavlink import mavutil
from simulations.config.logger import log_event

from simulations.config.drone_params import Drone8InchParams, DRONE_PARAMS

class Quad8InchSITL:
    def __init__(self, params: Drone8InchParams = DRONE_PARAMS,
                 bind_host: str = "127.0.0.1", bind_port: int = 14550):
        self.params = params
        self.bind_host = bind_host
        self.bind_port = bind_port
        
        # Estado 6-DOF del dron
        # Posición NED (x=Norte, y=Este, z=Abajo)
        self.pos = np.array([0.0, 0.0, 0.0], dtype=np.float64) # m
        self.vel = np.array([0.0, 0.0, 0.0], dtype=np.float64) # m/s
        self.acc = np.array([0.0, 0.0, 0.0], dtype=np.float64) # m/s^2
        
        # Actitud [roll, pitch, yaw] en radianes
        self.euler = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.omega = np.array([0.0, 0.0, 0.0], dtype=np.float64) # rad/s
        
        # Estado de piloto automático
        self.armed = False
        self.flight_mode = "STABILIZE"
        self.target_vel_body = np.zeros(3, dtype=np.float64) # [vx, vy, vz]
        self.target_yaw_rate = 0.0
        self.battery_pct = 100.0
        
        # Viento y perturbaciones
        self.wind_ned = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        
        # Comunicación MAVLink UDP
        self.sock = None
        self.client_addr = None
        self.mav = None
        self.running = False
        self.sim_thread = None
        self.mav_rx_thread = None
        self.state_lock = threading.Lock()

    def reset(self, initial_pos_ned: np.ndarray, initial_yaw_deg: float = 0.0):
        """Reinicia el estado del cuadricóptero a una posición dada."""
        with self.state_lock:
            self.pos = np.array(initial_pos_ned, dtype=np.float64)
            self.vel = np.zeros(3, dtype=np.float64)
            self.acc = np.zeros(3, dtype=np.float64)
            self.euler = np.array([0.0, 0.0, np.radians(initial_yaw_deg)], dtype=np.float64)
            self.omega = np.zeros(3, dtype=np.float64)
            self.target_vel_body = np.zeros(3, dtype=np.float64)
            self.target_yaw_rate = 0.0

    def start(self):
        """Inicia el motor de física y el servidor MAVLink."""
        if self.running:
            return
        
        # Crear socket UDP para MAVLink
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.bind_host, self.bind_port))
        self.sock.settimeout(0.05)
        
        # Objeto generador MAVLink
        self.mav = mavutil.mavlink.MAVLink(self, srcSystem=1, srcComponent=1)

        self.running = True
        self.sim_thread = threading.Thread(target=self._physics_loop, daemon=True)
        self.sim_thread.start()

        self.mav_rx_thread = threading.Thread(target=self._mavlink_rx_loop, daemon=True)
        self.mav_rx_thread.start()
        print(f"[SITL ARDUPILOT] Servidor MAVLink activo en {self.bind_host}:{self.bind_port}")

    def write(self, buf):
        """Método de escritura requerido por el generador MAVLink para enviar paquetes UDP."""
        if self.client_addr and self.sock:
            try:
                self.sock.sendto(buf, self.client_addr)
            except Exception:
                pass

    def _physics_loop(self):
        """Bucle de integración física 6-DOF a 100 Hz."""
        dt = 0.01
        last_telem_time = time.time()
        
        while self.running:
            t0 = time.time()
            with self.state_lock:
                # 1. Controlador de Vuelo Interno (Emulación de lazos de velocidad/actitud de ArduPilot)
                r_b2w = R_scipy.from_euler('xyz', self.euler, degrees=False)
                R_body_to_world = r_b2w.as_matrix()
                R_world_to_body = R_body_to_world.T

                if self.armed:
                    if self.flight_mode == "GUIDED":
                        # Convertir velocidad comandada en el marco del cuerpo a error y aceleración deseada
                        vel_body_current = R_world_to_body @ self.vel
                        vel_err_body = self.target_vel_body - vel_body_current

                        # Inclinación deseada en Roll y Pitch proporcional al error de velocidad
                        target_pitch = float(np.clip(-vel_err_body[0] * 0.25, -np.radians(self.params.max_tilt_angle_deg), np.radians(self.params.max_tilt_angle_deg)))
                        target_roll = float(np.clip(vel_err_body[1] * 0.25, -np.radians(self.params.max_tilt_angle_deg), np.radians(self.params.max_tilt_angle_deg)))
                        
                        # Empuje vertical para contrarrestar gravedad y alcanzar vz deseada
                        g = 9.81
                        thrust_z = -self.params.mass_kg * (g - (vel_err_body[2] * 2.0))
                    
                    elif self.flight_mode == "LAND":
                        target_roll = 0.0
                        target_pitch = 0.0
                        thrust_z = -self.params.mass_kg * (9.81 - 0.4) # Descenso constante a ~0.4 m/s
                        self.target_yaw_rate = 0.0
                    else: # STABILIZE / LOITER
                        target_roll = 0.0
                        target_pitch = 0.0
                        thrust_z = -self.params.mass_kg * 9.81
                        self.target_yaw_rate = 0.0

                    # Dinámica de orientación (filtro de primer orden / respuesta rápida de giroscopios)
                    tau_att = 0.08 # Constante de tiempo de respuesta de inclinación (80 ms para quad de 8")
                    self.euler[0] += (target_roll - self.euler[0]) * (dt / tau_att)
                    self.euler[1] += (target_pitch - self.euler[1]) * (dt / tau_att)
                    self.euler[2] += self.target_yaw_rate * dt
                    # Normalizar yaw entre [-pi, pi]
                    self.euler[2] = (self.euler[2] + np.pi) % (2 * np.pi) - np.pi

                    # 2. Fuerzas en el marco World NED
                    # Fuerza de empuje de los 4 rotores a lo largo de -Z_body
                    f_thrust_body = np.array([0.0, 0.0, thrust_z])
                    f_thrust_world = R_body_to_world @ f_thrust_body

                    # Gravedad
                    f_gravity_world = np.array([0.0, 0.0, self.params.mass_kg * 9.81])

                    # Resistencia aerodinámica (Drag) con viento
                    rel_vel = self.vel - self.wind_ned
                    f_drag_world = -0.5 * 1.225 * self.params.drag_coeff_xy * np.linalg.norm(rel_vel) * rel_vel

                    total_force = f_thrust_world + f_gravity_world + f_drag_world
                    self.acc = total_force / self.params.mass_kg

                    # Integración Euler
                    self.vel += self.acc * dt
                    self.pos += self.vel * dt

                    # Detección de contacto con el suelo a través de las patas de 10 cm
                    ground_z = -self.params.landing_gear_height_m
                    if self.pos[2] >= ground_z:
                        self.pos[2] = ground_z
                        self.vel = np.zeros(3)
                        self.acc = np.zeros(3)
                        if self.flight_mode == "LAND":
                            self.armed = False # Desarmado automático al tocar tierra en LAND
                else:
                    # Desarmado apoyado sobre las patas en el suelo
                    ground_z = -self.params.landing_gear_height_m
                    if self.pos[2] >= ground_z:
                        self.pos[2] = ground_z
                        self.vel = np.zeros(3)
                        self.acc = np.zeros(3)
                        self.euler[0] = 0.0
                        self.euler[1] = 0.0

            # 3. Transmisión de telemetría MAVLink periódica (50 Hz)
            now = time.time()
            if now - last_telem_time >= 0.02:
                self._send_mavlink_telemetry()
                last_telem_time = now

            elapsed = time.time() - t0
            time.sleep(max(0.0, dt - elapsed))

    def _send_mavlink_telemetry(self):
        """Genera y transmite paquetes MAVLink a clientes conectados."""
        if not self.client_addr:
            return
        
        boot_ms = int(time.time() * 1000) & 0xFFFFFFFF
        with self.state_lock:
            pos = self.pos.copy()
            vel = self.vel.copy()
            euler = self.euler.copy()
            armed = self.armed
            flight_mode = self.flight_mode

        # 1. HEARTBEAT
        mode_map = {"STABILIZE": 0, "GUIDED": 4, "LOITER": 5, "LAND": 9}
        custom_mode = mode_map.get(flight_mode, 0)
        base_mode = mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
        if armed:
            base_mode |= mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED

        self.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_QUADROTOR,
            mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA,
            base_mode,
            custom_mode,
            mavutil.mavlink.MAV_STATE_ACTIVE if armed else mavutil.mavlink.MAV_STATE_STANDBY
        )

        # 2. LOCAL_POSITION_NED
        self.mav.local_position_ned_send(
            boot_ms,
            float(pos[0]), float(pos[1]), float(pos[2]),
            float(vel[0]), float(vel[1]), float(vel[2])
        )

        # 3. ATTITUDE
        self.mav.attitude_send(
            boot_ms,
            float(euler[0]), float(euler[1]), float(euler[2]),
            float(self.omega[0]), float(self.omega[1]), float(self.omega[2])
        )

        # 4. SYS_STATUS
        self.mav.sys_status_send(
            0, 0, 0, 500, 15800, 1500, 95, 0, 0, 0, 0, 0, 0
        )

    def _mavlink_rx_loop(self):
        """Escucha comandos MAVLink entrantes desde la Radxa Zero 3W o GCS."""
        while self.running:
            try:
                data, addr = self.sock.recvfrom(2048)
                if not data:
                    continue
                self.client_addr = addr # Guardar dirección del cliente para responder
                
                # Decodificar mensajes MAVLink
                msgs = self.mav.parse_buffer(data)
                if msgs:
                    for msg in msgs:
                        self._handle_incoming_mavlink_msg(msg)
            except socket.timeout:
                pass
            except Exception as e:
                time.sleep(0.01)

    def _handle_incoming_mavlink_msg(self, msg):
        msg_type = msg.get_type()
        
        with self.state_lock:
            if msg_type == 'COMMAND_LONG':
                command = msg.command
                if command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
                    self.armed = (msg.param1 == 1.0)
                    log_event(f"[SITL] Motores {'ARMADOS' if self.armed else 'DESARMADOS'}")
                elif command == mavutil.mavlink.MAV_CMD_DO_SET_MODE:
                    mode_map = {0: "STABILIZE", 4: "GUIDED", 5: "LOITER", 9: "LAND"}
                    new_mode = mode_map.get(int(msg.param2), "GUIDED")
                    if self.flight_mode != new_mode:
                        self.flight_mode = new_mode
                        log_event(f"[SITL] Modo cambiado a: {self.flight_mode}")

            elif msg_type == 'SET_MODE':
                mode_map = {0: "STABILIZE", 4: "GUIDED", 5: "LOITER", 9: "LAND"}
                new_mode = mode_map.get(int(msg.custom_mode), "GUIDED")
                if self.flight_mode != new_mode:
                    self.flight_mode = new_mode
                    log_event(f"[SITL] Modo cambiado a: {self.flight_mode}")

            elif msg_type == 'SET_POSITION_TARGET_LOCAL_NED':
                if msg.coordinate_frame == mavutil.mavlink.MAV_FRAME_BODY_NED:
                    self.target_vel_body = np.array([msg.vx, msg.vy, msg.vz], dtype=np.float64)
                    self.target_yaw_rate = float(msg.yaw_rate)

    def get_state(self) -> Dict[str, Any]:
        """Devuelve una instantánea del estado 6-DOF del cuadricóptero."""
        with self.state_lock:
            return {
                "pos_ned": self.pos.copy(),
                "vel_ned": self.vel.copy(),
                "euler_rad": self.euler.copy(),
                "altitude_m": float(-self.pos[2]),
                "armed": self.armed,
                "flight_mode": self.flight_mode,
                "target_vel_body": self.target_vel_body.copy()
            }

    def stop(self):
        """Detiene la simulación."""
        self.running = False
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
        log_event("[SITL] Simulación de física detenida.")
