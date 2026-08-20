"""
Simulador de Cámara Nadir Industrial OV9281 (1280x720 @ 120 FPS Global Shutter).
Renderiza en tiempo real la proyección 3D del objetivo visual sobre el plano del sensor
a partir de la posición 6-DOF (x, y, z, roll, pitch, yaw) del cuadricóptero.
"""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
from typing import Tuple, Optional, Dict, Any

from simulations.config.camera_params import OV9281CameraParams, CAMERA_PARAMS
from simulations.vision.targets import VisualTargetGenerator

class SyntheticNadirCamera:
    def __init__(self, params: OV9281CameraParams = CAMERA_PARAMS,
                 target_type: str = "aruco", marker_id: int = 0,
                 marker_size_m: float = 0.055, pad_size_m: float = 0.80):
        self.params = params
        self.target_type = target_type
        self.marker_id = marker_id
        self.marker_size_m = marker_size_m
        self.pad_size_m = pad_size_m
        
        # Generar textura de la plataforma en alta resolución
        self.target_gen = VisualTargetGenerator(pad_size_m=pad_size_m, resolution_px_per_m=1500)
        self.pad_texture, self.pad_info = self.target_gen.create_skydock_landing_pad(
            target_type=target_type, marker_id=marker_id,
            marker_physical_size_m=marker_size_m, inner_marker_size_m=0.06
        )
        
        # Posición del landing pad en el mundo (NED: x, y, z=0)
        self.pad_world_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        
        # Generar textura de fondo de suelo (concreto rugoso / asfalto)
        self.ground_bg = self._create_ground_texture(width=1600, height=1200)
        
        # Matriz intrínseca y distorsión
        self.K = self.params.camera_matrix
        self.D = self.params.distortion_coeffs
        self.R_body_to_cam = self.params.R_body_to_cam
        self.mount_offset = self.params.mount_offset_frd_m

    def _create_ground_texture(self, width: int, height: int) -> np.ndarray:
        """Crea una textura de fondo de pista/concreto monocromática."""
        np.random.seed(42)
        base = np.ones((height, width), dtype=np.uint8) * 140
        noise = np.random.normal(0, 8, (height, width)).astype(np.int16)
        ground = np.clip(base + noise, 0, 255).astype(np.uint8)
        
        # Añadir sutiles líneas de cuadrícula en el suelo
        grid_step = 100
        for x in range(0, width, grid_step):
            cv2.line(ground, (x, 0), (x, height), 125, 1)
        for y in range(0, height, grid_step):
            cv2.line(ground, (0, y), (width, y), 125, 1)
        return ground

    def set_pad_position(self, x: float, y: float, z: float = 0.0):
        """Define la posición del objetivo en el marco World NED."""
        self.pad_world_pos = np.array([x, y, z], dtype=np.float64)

    def render_frame(self, drone_pos_ned: np.ndarray, drone_euler_rad: np.ndarray,
                     add_sensor_noise: bool = True) -> np.ndarray:
        """
        Renderiza un fotograma de 1280x720 píxeles correspondiente a la vista de la cámara OV9281.
        
        :param drone_pos_ned: [x_north, y_east, z_down] en metros. (z_down es negativo al volar).
        :param drone_euler_rad: [roll, pitch, yaw] en radianes.
        :param add_sensor_noise: Si se añade ruido gaussiano característico de sensor industrial.
        :return: Imagen monocromática (1280x720, uint8).
        """
        # 1. Calcular posición de la cámara en el mundo NED
        # Rotación del cuerpo respecto al mundo
        r_b2w = R_scipy.from_euler('xyz', drone_euler_rad, degrees=False)
        R_body_to_world = r_b2w.as_matrix()
        
        cam_pos_world = drone_pos_ned + R_body_to_world @ self.mount_offset
        altitude = -cam_pos_world[2] # Altitud sobre el suelo
        
        # 2. Rotación del mundo a la cámara óptica
        # R_world_to_cam = R_body_to_cam * R_world_to_body = R_body_to_cam * R_body_to_world.T
        R_world_to_cam = self.R_body_to_cam @ R_body_to_world.T
        
        # 3. Lienzo base de la cámara (gris ambiental)
        frame = np.ones((self.params.height, self.params.width), dtype=np.uint8) * 135
        
        # Si el dron está bajo tierra o altitud inválida, devolver frame oscuro
        if altitude <= 0.02:
            return np.ones((self.params.height, self.params.width), dtype=np.uint8) * 40

        # 4. Definir las 4 esquinas del Landing Pad en coordenadas de Mundo NED
        # Pad centrado en pad_world_pos, plano Z = pad_world_pos[2]
        half_w = self.pad_size_m / 2.0
        px, py, pz = self.pad_world_pos
        
        corners_world = np.array([
            [px - half_w, py - half_w, pz], # Top-Left (North-West)
            [px + half_w, py - half_w, pz], # Top-Right (North-East)
            [px + half_w, py + half_w, pz], # Bottom-Right (South-East)
            [px - half_w, py + half_w, pz]  # Bottom-Left (South-West)
        ], dtype=np.float64)

        # 5. Proyectar esquinas del pad a coordenadas de cámara y luego a píxeles
        corners_cam = (R_world_to_cam @ (corners_world - cam_pos_world).T).T # (4, 3)
        
        # Verificar que todos los puntos estén delante del sensor (Z_c > 0)
        if np.all(corners_cam[:, 2] > 0.01):
            # Proyección perspectiva usando la matriz K
            uv_homo = (self.K @ corners_cam.T).T # (4, 3)
            corners_img_px = (uv_homo[:, :2] / uv_homo[:, 2:3]).astype(np.float32) # (4, 2)
            
            # Coordenadas fuente correspondientes a la textura de alta resolución del pad
            tex_dim = self.pad_texture.shape[0]
            src_corners = np.array([
                [0, 0],
                [tex_dim, 0],
                [tex_dim, tex_dim],
                [0, tex_dim]
            ], dtype=np.float32)
            
            # Calcular matriz de homografía perspectiva
            H = cv2.getPerspectiveTransform(src_corners, corners_img_px)
            
            # Proyectar textura del pad sobre el lienzo de la cámara
            warped_pad = cv2.warpPerspective(
                self.pad_texture, H, (self.params.width, self.params.height),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
            
            # Crear máscara del pad proyectado para composición
            mask = cv2.warpPerspective(
                np.ones_like(self.pad_texture, dtype=np.uint8) * 255, H,
                (self.params.width, self.params.height),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
            
            # Componer el pad sobre el fondo
            frame = np.where(mask > 0, warped_pad, frame)

        # 6. Simulación de sensor OV9281 (Global Shutter, ruido, vignetting sutil)
        if add_sensor_noise:
            noise = np.random.normal(0, 2.5, frame.shape).astype(np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return frame
