"""
Renderizador 3D en Tercera Persona (Chase Cam & Orbit Cam) para el Cuadricóptero de 8 Pulgadas.
Genera una proyección 3D en perspectiva del entorno, la plataforma de aterrizaje SkyDock,
el cuadricóptero con sus 4 brazos/rotores orientados según Roll/Pitch/Yaw, su sombra y el cono de visión (FOV) de la cámara OV9281.
"""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
from typing import Tuple, Optional, List, Dict, Any
from simulations.vision.drone_mesh_loader import BumblebeeMeshModel

class ThirdPersonViewer3D:
    def __init__(self, width: int = 860, height: int = 520, fov_deg: float = 65.0):
        self.width = width
        self.height = height
        self.fov_rad = np.radians(fov_deg)
        
        # Parámetros intrínsecos de la cámara virtual 3D
        self.f = (self.width / 2.0) / np.tan(self.fov_rad / 2.0)
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0
        
        # Historial de trayectoria 3D
        self.path_history: List[np.ndarray] = []
        self.max_path_points = 250
        
        # Modelo 3D CAD Bumblebee
        self.cad_model = BumblebeeMeshModel()
        
        # Modo de cámara: "CHASE" (sigue al dron desde atrás) o "ORBIT" (punto fijo mirando a la plataforma)
        self.cam_mode = "CHASE"
        self.orbit_yaw_deg = 45.0
        self.orbit_pitch_deg = 28.0
        self.orbit_dist_m = 5.5

    def set_camera_mode(self, mode: str):
        if mode in ["CHASE", "ORBIT"]:
            self.cam_mode = mode

    def toggle_camera_mode(self):
        self.cam_mode = "ORBIT" if self.cam_mode == "CHASE" else "CHASE"

    def _project_point(self, pt_world: np.ndarray, R_w2c: np.ndarray, p_cam_world: np.ndarray) -> Optional[Tuple[int, int]]:
        """Proyecta un punto 3D del mundo (NED) al plano 2D de la pantalla."""
        # Vector relativo en marco de cámara 3D
        pt_rel = pt_world - p_cam_world
        p_c = R_w2c @ pt_rel # [x_right, y_down, z_forward]
        
        if p_c[2] <= 0.1: # Detrás de la cámara
            return None
        
        u = int(self.cx + (self.f * p_c[0] / p_c[2]))
        v = int(self.cy + (self.f * p_c[1] / p_c[2]))
        return (u, v)

    def render(self, drone_pos_ned: np.ndarray, drone_euler_rad: np.ndarray,
               pad_world_pos: np.ndarray = np.array([0.0, 0.0, 0.0]),
               target_detected: bool = False) -> np.ndarray:
        """
        Renderiza la escena 3D completa en tercera persona.
        """
        canvas = np.ones((self.height, self.width, 3), dtype=np.uint8) * 22 # Fondo oscuro
        
        # Actualizar historial de trayectoria
        self.path_history.append(drone_pos_ned.copy())
        if len(self.path_history) > self.max_path_points:
            self.path_history.pop(0)

        # 1. Definir posición y orientación de la Cámara Virtual 3D
        drone_x, drone_y, drone_z = drone_pos_ned
        drone_yaw = drone_euler_rad[2]

        if self.cam_mode == "CHASE":
            # Cámara detrás y arriba del dron siguiendo su orientación
            dist_behind = 2.4
            height_above = 1.1
            
            # Posición de la cámara en el mundo NED
            cam_offset_world = np.array([
                -dist_behind * np.cos(drone_yaw),
                -dist_behind * np.sin(drone_yaw),
                -height_above
            ])
            p_cam_world = drone_pos_ned + cam_offset_world
            
            # Punto al que mira la cámara: Centro del dron o ligeramente adelante
            look_at = drone_pos_ned + np.array([0.5 * np.cos(drone_yaw), 0.5 * np.sin(drone_yaw), 0.1])
        else: # ORBIT
            # Cámara orbital fija mirando al punto medio entre el dron y el landing pad
            rad_yaw = np.radians(self.orbit_yaw_deg)
            rad_pitch = np.radians(self.orbit_pitch_deg)
            
            p_cam_world = np.array([
                -self.orbit_dist_m * np.cos(rad_pitch) * np.cos(rad_yaw),
                -self.orbit_dist_m * np.cos(rad_pitch) * np.sin(rad_yaw),
                -self.orbit_dist_m * np.sin(rad_pitch)
            ])
            look_at = np.array([drone_x * 0.4, drone_y * 0.4, 0.0])

        # Construir matriz de vista LookAt
        forward = look_at - p_cam_world
        forward = forward / np.linalg.norm(forward)
        
        # En NED: Arriba es -Z
        world_up = np.array([0.0, 0.0, -1.0])
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-4:
            right = np.array([0.0, 1.0, 0.0])
        else:
            right = right / np.linalg.norm(right)
        
        down = np.cross(forward, right)
        # R_w2c transforma puntos del mundo a [right, down, forward] de la cámara
        R_w2c = np.array([right, down, forward])

        # 2. Dibujar Plano del Suelo y Cuadrícula (Z = 0)
        grid_range = 4.0
        grid_step = 0.5
        
        # Líneas de cuadrícula en el suelo
        for x in np.arange(-grid_range, grid_range + 0.01, grid_step):
            p1 = self._project_point(np.array([x, -grid_range, 0.0]), R_w2c, p_cam_world)
            p2 = self._project_point(np.array([x, grid_range, 0.0]), R_w2c, p_cam_world)
            if p1 and p2:
                color = (65, 65, 65) if abs(x) > 0.01 else (100, 100, 100)
                cv2.line(canvas, p1, p2, color, 1)

        for y in np.arange(-grid_range, grid_range + 0.01, grid_step):
            p1 = self._project_point(np.array([-grid_range, y, 0.0]), R_w2c, p_cam_world)
            p2 = self._project_point(np.array([grid_range, y, 0.0]), R_w2c, p_cam_world)
            if p1 and p2:
                color = (65, 65, 65) if abs(y) > 0.01 else (100, 100, 100)
                cv2.line(canvas, p1, p2, color, 1)

        # 3. Dibujar Plataforma SkyDock en 3D en el suelo
        pad_size = 0.80
        hw = pad_size / 2.0
        px, py, pz = pad_world_pos
        pad_corners = [
            np.array([px - hw, py - hw, pz]),
            np.array([px + hw, py - hw, pz]),
            np.array([px + hw, py + hw, pz]),
            np.array([px - hw, py + hw, pz])
        ]
        pad_pts_2d = [self._project_point(pt, R_w2c, p_cam_world) for pt in pad_corners]
        if all(pt is not None for pt in pad_pts_2d):
            pts_arr = np.array(pad_pts_2d, dtype=np.int32)
            cv2.fillPoly(canvas, [pts_arr], (70, 70, 70))
            cv2.polylines(canvas, [pts_arr], True, (0, 220, 220), 2)
            
            # Helipad Circle en 3D
            circle_pts = []
            for deg in range(0, 360, 15):
                rad = np.radians(deg)
                c_pt = np.array([px + 0.32 * np.cos(rad), py + 0.32 * np.sin(rad), pz])
                proj = self._project_point(c_pt, R_w2c, p_cam_world)
                if proj:
                    circle_pts.append(proj)
            if len(circle_pts) > 10:
                cv2.polylines(canvas, [np.array(circle_pts, dtype=np.int32)], True, (255, 255, 255), 1)

        # 4. Dibujar Trayectoria 3D
        if len(self.path_history) > 1:
            for i in range(1, len(self.path_history)):
                pt1 = self._project_point(self.path_history[i-1], R_w2c, p_cam_world)
                pt2 = self._project_point(self.path_history[i], R_w2c, p_cam_world)
                if pt1 and pt2:
                    # Degradado de color
                    alpha = i / len(self.path_history)
                    color = (int(0 * alpha), int(140 * alpha), int(255 * alpha))
                    cv2.line(canvas, pt1, pt2, color, 2)

        # 5. Sombra del Dron y Línea de Altura Vertical
        ground_shadow = np.array([drone_x, drone_y, 0.0])
        p_drone_2d = self._project_point(drone_pos_ned, R_w2c, p_cam_world)
        p_shadow_2d = self._project_point(ground_shadow, R_w2c, p_cam_world)

        if p_drone_2d and p_shadow_2d:
            # Línea vertical punteada indicadora de altitud
            cv2.line(canvas, p_shadow_2d, p_drone_2d, (120, 120, 120), 1, cv2.LINE_AA)
            # Sombra elíptica
            cv2.circle(canvas, p_shadow_2d, max(4, int(12 - drone_pos_ned[2] * 2)), (35, 35, 35), -1)

        # 6. Cono de Visión Nadir OV9281 (Frustum de Cámara hacia abajo)
        # Apertura de la cámara ~75 grados proyectada en el suelo
        fov_cone_half = 0.65 * max(0.2, -drone_pos_ned[2])
        cone_corners_body = [
            np.array([-fov_cone_half, -fov_cone_half, -drone_pos_ned[2]]),
            np.array([ fov_cone_half, -fov_cone_half, -drone_pos_ned[2]]),
            np.array([ fov_cone_half,  fov_cone_half, -drone_pos_ned[2]]),
            np.array([-fov_cone_half,  fov_cone_half, -drone_pos_ned[2]])
        ]
        
        r_b2w = R_scipy.from_euler('xyz', drone_euler_rad, degrees=False)
        R_body_to_world = r_b2w.as_matrix()
        
        cone_pts_world = [drone_pos_ned + R_body_to_world @ c for c in cone_corners_body]
        # Forzar Z en el suelo Z=0
        for cp in cone_pts_world:
            cp[2] = 0.0

        cone_pts_2d = [self._project_point(cp, R_w2c, p_cam_world) for cp in cone_pts_world]
        if all(pt is not None for pt in cone_pts_2d) and p_drone_2d:
            cone_color = (0, 200, 100) if target_detected else (180, 120, 0)
            # Dibujar 4 líneas del cono de la cámara
            for cpt in cone_pts_2d:
                cv2.line(canvas, p_drone_2d, cpt, cone_color, 1, cv2.LINE_AA)
            cv2.polylines(canvas, [np.array(cone_pts_2d, dtype=np.int32)], True, cone_color, 1)

        # 7. Renderizado del Modelo 3D CAD BUMBLEBEE
        rendered_cad = False
        if self.cad_model.is_loaded:
            rendered_cad = self.cad_model.render(
                canvas=canvas,
                drone_pos_ned=drone_pos_ned,
                R_body_to_world=R_body_to_world,
                R_w2c=R_w2c,
                p_cam_world=p_cam_world,
                f_px=self.f
            )

        # 8. Motores y Rotores (Frontales en Naranja/Rojo, Traseros en Verde)
        arm_len = 0.185
        motor_pos_body = [
            np.array([ arm_len * 0.707,  arm_len * 0.707, -0.01]), # Front-Right
            np.array([-arm_len * 0.707,  arm_len * 0.707, -0.01]), # Rear-Right
            np.array([-arm_len * 0.707, -arm_len * 0.707, -0.01]), # Rear-Left
            np.array([ arm_len * 0.707, -arm_len * 0.707, -0.01])  # Front-Left
        ]
        
        motor_pts_world = [drone_pos_ned + R_body_to_world @ m for m in motor_pos_body]
        motor_pts_2d = [self._project_point(mp, R_w2c, p_cam_world) for mp in motor_pts_world]

        if p_drone_2d and all(pt is not None for pt in motor_pts_2d):
            # Si no se pudo cargar el CAD, dibujar fallback paramétrico
            if not rendered_cad:
                cv2.line(canvas, motor_pts_2d[0], motor_pts_2d[2], (180, 180, 180), 3)
                cv2.line(canvas, motor_pts_2d[3], motor_pts_2d[1], (180, 180, 180), 3)
                leg_len_m = 0.10
                for i in range(4):
                    leg_bottom_body = motor_pos_body[i] + np.array([0.0, 0.0, leg_len_m])
                    leg_bottom_world = drone_pos_ned + R_body_to_world @ leg_bottom_body
                    p_foot_2d = self._project_point(leg_bottom_world, R_w2c, p_cam_world)
                    if p_foot_2d and motor_pts_2d[i]:
                        cv2.line(canvas, motor_pts_2d[i], p_foot_2d, (140, 140, 140), 2)
                        cv2.circle(canvas, p_foot_2d, 3, (200, 200, 200), -1)

            # Rotores / Hélices 8 pulgadas
            rotor_radius_px = 12
            cv2.circle(canvas, motor_pts_2d[0], rotor_radius_px, (0, 100, 255), 2)
            cv2.circle(canvas, motor_pts_2d[3], rotor_radius_px, (0, 100, 255), 2)
            cv2.circle(canvas, motor_pts_2d[1], rotor_radius_px, (0, 230, 0), 2)
            cv2.circle(canvas, motor_pts_2d[2], rotor_radius_px, (0, 230, 0), 2)

            # Nariz / Flecha de Heading Frontal
            nose_body = np.array([0.16, 0.0, -0.01])
            nose_world = drone_pos_ned + R_body_to_world @ nose_body
            p_nose_2d = self._project_point(nose_world, R_w2c, p_cam_world)
            if p_nose_2d:
                cv2.line(canvas, p_drone_2d, p_nose_2d, (0, 255, 255), 2)

        # Overlay HUD del modo de cámara 3D
        cam_label = f"3D CHASE CAM (BUMBLEBEE CAD)" if self.cam_mode == "CHASE" else "3D ORBIT CAM (BUMBLEBEE CAD)"
        cv2.putText(canvas, cam_label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(canvas, "Tecla [C]: Alternar Chase / Orbit Cam", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)

        return canvas
