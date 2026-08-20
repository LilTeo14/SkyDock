"""
Controlador de Guiado y Aterrizaje Autónomo de Precisión (PID + Máquina de Estados).
Calcula comandos de velocidad MAVLink (SET_POSITION_TARGET_LOCAL_NED) a partir de los
errores de posición visual $(e_x, e_y, e_z)$ obtenidos de la cámara OV9281.
"""
import numpy as np
import time
from enum import Enum
from typing import Tuple, Dict, Any, Optional
from simulations.config.logger import log_event

class LandingState(Enum):
    IDLE = "IDLE"
    TAKEOFF = "TAKEOFF"
    SEARCHING = "SEARCHING"
    ALIGNING = "ALIGNING"
    DESCENDING = "DESCENDING"
    TOUCHDOWN = "TOUCHDOWN"
    ABORT = "ABORT"

class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, max_out: float, max_integral: float = 1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_out = max_out
        self.max_integral = max_integral
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = 0.0

    def update(self, error: float, dt: float) -> float:
        if dt <= 0.0:
            return 0.0
        
        # Integral con anti-windup
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.max_integral, self.max_integral)
        
        # Derivativo
        derivative = (error - self.prev_error) / dt if self.prev_time > 0 else 0.0
        self.prev_error = error
        self.prev_time = time.time()
        
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        return float(np.clip(output, -self.max_out, self.max_out))
        
class SmoothPositionController:
    """
    Controlador de posición suavizado con perfil de desaceleración tipo ArduPilot (Sqrt Controller),
    amortiguamiento por velocidad estimada (Kalman), integrador de compensación de viento/perturbación
    y limitador de aceleración (Slew Rate).
    Elimina sobreimpulsos (overshoot) y oscilaciones cíclicas.
    """
    def __init__(self, kp_pos: float = 0.90, kd_vel: float = 0.30, ki_bias: float = 0.08, max_vel: float = 1.1, max_accel: float = 1.3):
        self.kp_pos = kp_pos
        self.kd_vel = kd_vel
        self.ki_bias = ki_bias
        self.max_vel = max_vel
        self.max_accel = max_accel
        self.last_cmd = 0.0
        self.integral_bias = 0.0

    def reset(self):
        self.last_cmd = 0.0
        self.integral_bias = 0.0

    def update(self, pos_error: float, current_vel: float, dt: float) -> float:
        if dt <= 0.0:
            return self.last_cmd

        dist = abs(pos_error)
        
        # 1. Perfil de frenado suave de raíz cuadrada (evita sobrepaso)
        v_brake = np.sqrt(2.0 * self.max_accel * max(0.005, dist))
        v_target_mag = min(self.max_vel, v_brake, self.kp_pos * dist)
        v_des = np.sign(pos_error) * v_target_mag

        # 2. Integrador de viento/bias en zona cercana (< 0.5m) con anti-windup
        if dist < 0.5:
            self.integral_bias += pos_error * dt
            self.integral_bias = float(np.clip(self.integral_bias, -0.3, 0.3))
        else:
            self.integral_bias *= 0.95 # Leaky decay fuera de aproximación fina

        # 3. Amortiguamiento Damping con velocidad Kalman + Compensación de viento
        v_out = v_des - (self.kd_vel * current_vel) + (self.ki_bias * self.integral_bias)
        v_out = float(np.clip(v_out, -self.max_vel, self.max_vel))

        # 4. Limitador de rampa de aceleración (Slew-rate limiter)
        max_delta = self.max_accel * dt
        delta = np.clip(v_out - self.last_cmd, -max_delta, max_delta)
        self.last_cmd = float(self.last_cmd + delta)
        return self.last_cmd


class AutonomousPrecisionLandingController:
    def __init__(self):
        # Controladores de posición suavizados para ejes horizontales (X, Y)
        self.pos_ctrl_x = SmoothPositionController(kp_pos=0.90, kd_vel=0.30, ki_bias=0.08, max_vel=1.1, max_accel=1.3)
        self.pos_ctrl_y = SmoothPositionController(kp_pos=0.90, kd_vel=0.30, ki_bias=0.08, max_vel=1.1, max_accel=1.3)
        
        # Controlador de guiñada suave
        self.pid_yaw = PIDController(kp=1.2, ki=0.0, kd=0.15, max_out=0.6) # rad/s
        
        # Estado actual
        self.state = LandingState.IDLE
        self.state_start_time = time.time()
        self.last_update_time = time.time()
        
        # Parámetros de umbral
        self.align_tolerance_m = 0.15      # Error horizontal admisible para descender (15 cm)
        self.touchdown_altitude_m = 0.12   # Altitud crítica de contacto con el suelo (12 cm)
        self.target_descent_rate_max = 0.55 # Velocidad máxima de descenso (m/s)
        self.target_descent_rate_min = 0.15 # Velocidad mínima de descenso al aproximar
        self.abort_timeout_s = 3.0         # Tiempo sin detección para abortar si estamos bajos

    def set_state(self, new_state: LandingState):
        if self.state != new_state:
            log_event(f"[RADXA CONTROLLER] Transición de estado: {self.state.value} -> {new_state.value}")
            self.state = new_state
            self.state_start_time = time.time()
            if new_state == LandingState.ALIGNING:
                self.pos_ctrl_x.reset()
                self.pos_ctrl_y.reset()
                self.pid_yaw.reset()

    def update(self, vision_data: Dict[str, Any], current_altitude: float) -> Tuple[np.ndarray, float, LandingState]:
        """
        Ejecuta el ciclo de control.
        
        :param vision_data: Datos del detector (pos_body_filtered, detected, kalman_active, yaw_error_rad, etc.)
        :param current_altitude: Altitud estimada del dron (m).
        :return: (vel_cmd_body [vx, vy, vz], yaw_rate_cmd_rad_s, current_state)
        """
        now = time.time()
        dt = now - self.last_update_time if self.last_update_time > 0 else 0.033
        self.last_update_time = now

        target_detected = vision_data.get("kalman_active", False)
        pos_body = vision_data.get("pos_body_filtered") # [dx_forward, dy_right, dz_down]
        vel_body = vision_data.get("vel_body_filtered") # [vx_forward, vy_right, vz_down]
        yaw_err = vision_data.get("yaw_error_rad", 0.0) # Error angular respecto a la plataforma
        
        current_vx = vel_body[0] if vel_body is not None else 0.0
        current_vy = vel_body[1] if vel_body is not None else 0.0

        vel_cmd_body = np.zeros(3, dtype=np.float64) # [vx_forward, vy_right, vz_down]
        yaw_rate_cmd = 0.0

        # --- MÁQUINA DE ESTADOS ---
        if self.state == LandingState.IDLE:
            # Esperando inicio de misión
            vel_cmd_body = np.zeros(3)
            yaw_rate_cmd = 0.0
            self.pos_ctrl_x.reset()
            self.pos_ctrl_y.reset()
            self.pid_yaw.reset()

        elif self.state == LandingState.TAKEOFF:
            # Subir a altitud de búsqueda (ej. 3.0 m)
            if current_altitude >= 2.5:
                self.set_state(LandingState.SEARCHING)
            else:
                vel_cmd_body = np.array([0.0, 0.0, -0.8]) # Ascender (-Z en NED/Body)

        elif self.state == LandingState.SEARCHING:
            # Si se adquiere el objetivo, pasar a alinear
            if target_detected and pos_body is not None:
                self.set_state(LandingState.ALIGNING)
            else:
                # Si estamos muy bajos para ver la plataforma, ganar altitud para ampliar cono FOV
                if current_altitude < 2.6:
                    vel_cmd_body = np.array([0.0, 0.0, -0.6]) # Ascender suavemente a 2.8m
                else:
                    vel_cmd_body = np.zeros(3)

        elif self.state == LandingState.ALIGNING:
            if not target_detected or pos_body is None:
                # Si se pierde momentáneamente, decaer velocidad suavemente sin frenazos bruscos
                self.pos_ctrl_x.last_cmd *= 0.88
                self.pos_ctrl_y.last_cmd *= 0.88
                vel_cmd_body = np.array([self.pos_ctrl_x.last_cmd, self.pos_ctrl_y.last_cmd, 0.0])
                if (now - self.state_start_time) > self.abort_timeout_s:
                    self.set_state(LandingState.SEARCHING)
                return vel_cmd_body, yaw_rate_cmd, self.state

            err_x = pos_body[0] # Error hacia adelante (+X)
            err_y = pos_body[1] # Error hacia la derecha (+Y)
            horiz_err = np.hypot(err_x, err_y)

            # Comandar velocidades horizontales suaves con frenado Sqrt y amortiguamiento Kalman
            vx = self.pos_ctrl_x.update(err_x, current_vx, dt)
            vy = self.pos_ctrl_y.update(err_y, current_vy, dt)
            vz = 0.0 # Mantener altitud mientras se alinea
            yaw_rate_cmd = -self.pid_yaw.update(yaw_err, dt)

            vel_cmd_body = np.array([vx, vy, vz])

            # Si el error horizontal y angular son admisibles, comenzar descenso
            if horiz_err < self.align_tolerance_m and abs(yaw_err) < np.radians(8.0):
                self.set_state(LandingState.DESCENDING)

        elif self.state == LandingState.DESCENDING:
            if not target_detected or pos_body is None:
                self.pos_ctrl_x.last_cmd *= 0.88
                self.pos_ctrl_y.last_cmd *= 0.88
                vel_cmd_body = np.array([self.pos_ctrl_x.last_cmd, self.pos_ctrl_y.last_cmd, 0.15])
                if (now - self.state_start_time) > self.abort_timeout_s:
                    self.set_state(LandingState.ABORT)
                return vel_cmd_body, yaw_rate_cmd, self.state

            err_x = pos_body[0]
            err_y = pos_body[1]
            err_z = pos_body[2] # Distancia vertical hacia el suelo (+Z hacia abajo)
            horiz_err = np.hypot(err_x, err_y)

            # Centrado continuo horizontal amortiguado y corrección de guiñada
            vx = self.pos_ctrl_x.update(err_x, current_vx, dt)
            vy = self.pos_ctrl_y.update(err_y, current_vy, dt)
            yaw_rate_cmd = -self.pid_yaw.update(yaw_err, dt)

            # Velocidad de descenso variable en función del error horizontal y la altitud
            # Si el dron se desvía, frenar descenso para corregir
            centering_factor = np.clip(1.0 - (horiz_err / 0.35), 0.1, 1.0)
            
            # Velocidad de descenso proporcional a la altitud actual
            descent_speed = np.clip(
                self.target_descent_rate_max * (err_z / 2.0) * centering_factor,
                self.target_descent_rate_min,
                self.target_descent_rate_max
            )
            vz = descent_speed # +vz es hacia abajo en convención NED/Body

            vel_cmd_body = np.array([vx, vy, vz])

            # Umbral de contacto / touchdown
            if err_z <= self.touchdown_altitude_m or current_altitude <= self.touchdown_altitude_m:
                self.set_state(LandingState.TOUCHDOWN)

        elif self.state == LandingState.TOUCHDOWN:
            # Contacto final con la plataforma: corte de comandos de traslación
            vel_cmd_body = np.array([0.0, 0.0, 0.2]) # Suave empuje final hacia abajo
            yaw_rate_cmd = 0.0

        elif self.state == LandingState.ABORT:
            # Failsafe: trepar a altura de seguridad
            vel_cmd_body = np.array([0.0, 0.0, -1.0])
            if current_altitude >= 3.0:
                self.set_state(LandingState.SEARCHING)

        return vel_cmd_body, yaw_rate_cmd, self.state
