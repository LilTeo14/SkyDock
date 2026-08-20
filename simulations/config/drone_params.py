"""
Configuración y parámetros físicos para el cuadricóptero de 8 pulgadas.
"""
from dataclasses import dataclass, field
import numpy as np

@dataclass
class Drone8InchParams:
    # Dimensiones y masa
    frame_type: str = "QUAD_X"
    wheelbase_m: float = 0.370          # 370 mm diagonal (8 pulgadas)
    arm_length_m: float = 0.185         # Distancia eje motor al centro
    landing_gear_height_m: float = 0.10 # 10 cm de altura de patas (permite ver el QR al tocar tierra)
    mass_kg: float = 1.450              # Masa total (frame + electrónica + batería 6S + Radxa + cámara)
    
    # Inercias (kg*m^2)
    ixx: float = 0.0145
    iyy: float = 0.0145
    izz: float = 0.0260
    
    # Sistema de propulsión (Motores 2806.5 / 1300KV + Hélices 8x4.5")
    num_rotors: int = 4
    prop_diameter_m: float = 0.2032     # 8 pulgadas en metros
    prop_pitch_m: float = 0.1143        # 4.5 pulgadas en metros
    max_thrust_per_motor_n: float = 12.0 # ~1.22 kgf por motor (Total: 4.88 kgf -> TWR = ~3.37)
    hover_throttle_ratio: float = 0.30  # ~30% de acelerador para vuelo estacionario
    
    # Coeficientes aerodinámicos
    drag_coeff_xy: float = 0.35
    drag_coeff_z: float = 0.45
    
    # Límites de vuelo autónomo (ArduPilot GUIDED mode)
    max_velocity_xy_m_s: float = 12.0
    max_velocity_z_up_m_s: float = 3.5
    max_velocity_z_down_m_s: float = 2.0
    max_tilt_angle_deg: float = 35.0
    max_yaw_rate_deg_s: float = 120.0
    
    # Puertos y MAVLink
    sitl_udp_host: str = "127.0.0.1"
    sitl_udp_port: int = 14550          # Puerto estándar de conexión MAVLink
    companion_sysid: int = 1            # ID de la companion computer (o GCS)
    companion_compid: int = 191         # MAV_COMP_ID_ONBOARD_COMPUTER (191) o 1

# Instancia por defecto
DRONE_PARAMS = Drone8InchParams()
