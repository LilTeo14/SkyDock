"""
Parámetros de la cámara industrial AED7-720P Global Shutter (OmniVision OV9281).
Cámara montada en posición nadir (apuntando hacia abajo).
"""
from dataclasses import dataclass, field
import numpy as np

@dataclass
class OV9281CameraParams:
    # Identificación y sensor
    model_name: str = "AED7-720P Global Shutter USB Industrial Camera"
    sensor_model: str = "OmniVision OV9281"
    shutter_type: str = "GLOBAL_SHUTTER"
    pixel_format: str = "MONO8"         # Sensor monocromático ideal para tracking de alta velocidad
    
    # Resolución y tasa de cuadros
    width: int = 1280
    height: int = 720
    fps: int = 120                      # 120 FPS nativos de alta velocidad
    
    # Óptica y sensor físico
    sensor_width_mm: float = 3.84       # 1/4" sensor format
    sensor_height_mm: float = 2.16
    pixel_size_um: float = 3.0          # 3.0 µm x 3.0 µm BSI pixel
    lens_focal_length_mm: float = 2.8   # Lente gran angular estándar (~78° H-FOV)
    
    # Matriz intrínseca calibrada K:
    # fx = (focal_length_mm / sensor_width_mm) * width ~ 933.3 px
    # fy = (focal_length_mm / sensor_height_mm) * height ~ 933.3 px
    # cx = width / 2 = 640.0 px
    # cy = height / 2 = 360.0 px
    fx: float = 933.33
    fy: float = 933.33
    cx: float = 640.0
    cy: float = 360.0
    
    # Coeficientes de distorsión [k1, k2, p1, p2, k3]
    distortion_coeffs: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64))
    
    # Posición y orientación de montaje relativa al centro de masa del dron (Frame FRD: X=adelante, Y=derecha, Z=abajo)
    # Montaje nadir centrado: Cámara apunta verticalmente hacia abajo con línea de visión despejada
    mount_offset_frd_m: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float64))
    
    # Matriz de rotación cuerpo a cámara (Body FRD -> Camera Optical: X_cam=derecha, Y_cam=abajo_cuerpo/atrás, Z_cam=apuntando_al_suelo)
    # En cámara nadir:
    # X_cam = +Y_body (derecha)
    # Y_cam = -X_body (hacia atrás del dron para que el top de la imagen mire al frente del dron)
    # Z_cam = +Z_body (hacia abajo)
    R_body_to_cam: np.ndarray = field(default_factory=lambda: np.array([
        [0.0,  1.0,  0.0],
        [-1.0, 0.0,  0.0],
        [0.0,  0.0,  1.0]
    ], dtype=np.float64))

    @property
    def camera_matrix(self) -> np.ndarray:
        return np.array([
            [self.fx, 0.0,     self.cx],
            [0.0,     self.fy, self.cy],
            [0.0,     0.0,     1.0]
        ], dtype=np.float64)

CAMERA_PARAMS = OV9281CameraParams()
