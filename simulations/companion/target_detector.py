"""
Detector de Objetivos Visuales (ArUco / Código QR) y Estimador de Pose 6-DOF.
Diseñado para ejecutarse a bordo de la Radxa Zero 3W utilizando la cámara OV9281.
Incluye filtro de Kalman 3D para seguimiento continuo y predicción ante oclusión.
"""
import cv2
import numpy as np
import time
from scipy.spatial.transform import Rotation as R_scipy
from typing import Tuple, Optional, Dict, Any

from simulations.config.camera_params import OV9281CameraParams, CAMERA_PARAMS

class KalmanFilter3D:
    """Filtro de Kalman para posición 3D (X, Y, Z) y velocidades (Vx, Vy, Vz)."""
    def __init__(self, q_pos=0.01, q_vel=0.1, r_pos=0.05):
        self.state = np.zeros(6, dtype=np.float64) # [x, y, z, vx, vy, vz]
        self.P = np.eye(6, dtype=np.float64) * 1.0
        self.q_pos = q_pos
        self.q_vel = q_vel
        self.r_pos = r_pos
        self.is_initialized = False
        self.last_time = time.time()

    def reset(self, initial_pos: np.ndarray):
        self.state = np.zeros(6, dtype=np.float64)
        self.state[0:3] = initial_pos
        self.P = np.eye(6, dtype=np.float64) * 0.1
        self.is_initialized = True
        self.last_time = time.time()

    def predict(self, dt: float):
        if not self.is_initialized or dt <= 0:
            return
        # Matriz de transición de estado F
        F = np.eye(6, dtype=np.float64)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        
        # Ruido de proceso Q
        Q = np.zeros((6, 6), dtype=np.float64)
        for i in range(3):
            Q[i, i] = self.q_pos * dt
            Q[i + 3, i + 3] = self.q_vel * dt

        self.state = F @ self.state
        self.P = F @ self.P @ F.T + Q

    def update(self, z_pos: np.ndarray):
        if not self.is_initialized:
            self.reset(z_pos)
            return

        # Matriz de observación H (observamos solo posición x, y, z)
        H = np.zeros((3, 6), dtype=np.float64)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0

        # Ruido de medición R
        R = np.eye(3, dtype=np.float64) * self.r_pos

        y = z_pos - (H @ self.state) # Innovación
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S) # Ganancia de Kalman

        self.state = self.state + (K @ y)
        I_KH = np.eye(6, dtype=np.float64) - (K @ H)
        self.P = I_KH @ self.P

    @property
    def position(self) -> np.ndarray:
        return self.state[0:3]

    @property
    def velocity(self) -> np.ndarray:
        return self.state[3:6]


class VisualTargetDetector:
    def __init__(self, camera_params: OV9281CameraParams = CAMERA_PARAMS,
                 dict_type=cv2.aruco.DICT_4X4_50, marker_size_m: float = 0.055):
        self.params = camera_params
        self.K = camera_params.camera_matrix
        self.D = camera_params.distortion_coeffs
        self.R_body_to_cam = camera_params.R_body_to_cam
        self.R_cam_to_body = self.R_body_to_cam.T
        self.marker_size_m = marker_size_m
        
        # Detector ArUco
        self.dict_type = dict_type
        if hasattr(cv2.aruco, 'getPredefinedDictionary'):
            self.dictionary = cv2.aruco.getPredefinedDictionary(self.dict_type)
            self.detector_params = cv2.aruco.DetectorParameters()
            self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.detector_params)
        else:
            self.dictionary = cv2.aruco.Dictionary_get(self.dict_type)
            self.detector_params = cv2.aruco.DetectorParameters_create()
            self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self.detector = None

        # Filtro de Kalman 3D para la posición del objetivo en el marco del cuerpo (FRD)
        self.kf = KalmanFilter3D()
        self.last_detection_time = 0.0
        self.max_coast_time_s = 0.5 # Tiempo para mantener predicción si se pierde el marcador

    def detect(self, frame: np.ndarray, current_attitude_rad: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Procesa el fotograma mono de la OV9281 y extrae la pose del objetivo.
        
        :param frame: Fotograma monocromático (1280x720).
        :param current_attitude_rad: [roll, pitch, yaw] del dron para proyección a World NED.
        :return: Diccionario con resultados de detección, pose y estado de Kalman.
        """
        now = time.time()
        dt = now - self.kf.last_time if self.kf.last_time > 0 else 0.033
        self.kf.last_time = now
        self.kf.predict(dt)

        detected = False
        target_id = -1
        rvec = None
        tvec_cam = None
        tvec_body = None
        tvec_world = None
        corners = None
        euler_deg = None

        # 1. Detección ArUco
        if self.detector is not None:
            detected_corners, ids, _ = self.detector.detectMarkers(frame)
        else:
            detected_corners, ids, _ = cv2.aruco.detectMarkers(frame, self.dictionary, parameters=self.detector_params)

        if ids is not None and len(ids) > 0:
            detected = True
            self.last_detection_time = now
            target_id = int(np.array(ids).flatten()[0])
            corners = detected_corners[0]

            # 2. Estimación de Pose PnP
            # Puntos 3D del marcador en su propio sistema de coordenadas
            half_s = self.marker_size_m / 2.0
            obj_points = np.array([
                [-half_s,  half_s, 0.0],
                [ half_s,  half_s, 0.0],
                [ half_s, -half_s, 0.0],
                [-half_s, -half_s, 0.0]
            ], dtype=np.float64)

            success, rvec, tvec = cv2.solvePnP(
                obj_points, corners.reshape(-1, 2).astype(np.float64),
                self.K, self.D, flags=cv2.SOLVEPNP_IPPE_SQUARE
            )

            if success:
                tvec_cam = tvec.flatten() # [X_cam, Y_cam, Z_cam]
                
                # Transformar de Cámara Óptica a Drone Body FRD
                # P_body = R_cam_to_body * P_cam + mount_offset
                tvec_body = self.R_cam_to_body @ tvec_cam + self.params.mount_offset_frd_m
                
                # Desacoplamiento de Actitud: proyectar al plano horizontal nivelado con el rumbo del dron
                if current_attitude_rad is not None:
                    r_b2w = R_scipy.from_euler('xyz', current_attitude_rad, degrees=False)
                    R_body_to_world = r_b2w.as_matrix()
                    tvec_world_raw = R_body_to_world @ tvec_body
                    
                    yaw = current_attitude_rad[2]
                    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
                    R_yaw = np.array([
                        [cos_y, sin_y, 0.0],
                        [-sin_y, cos_y, 0.0],
                        [0.0, 0.0, 1.0]
                    ])
                    # Medición nivelada inmune a inclinaciones de Roll y Pitch
                    tvec_leveled = R_yaw @ tvec_world_raw
                    # Mantener Z como altitud vertical positiva hacia el suelo
                    tvec_leveled[2] = abs(tvec_world_raw[2])
                else:
                    tvec_leveled = tvec_body

                # Actualizar Filtro de Kalman con la medición nivelada
                self.kf.update(tvec_leveled)

                # Calcular ángulos de Euler y error de guiñada del marcador relativo al dron
                rmat, _ = cv2.Rodrigues(rvec)
                R_m2b = self.R_cam_to_body @ rmat
                rot_m2b = R_scipy.from_matrix(R_m2b)
                yaw_rel_deg = float(rot_m2b.as_euler('zyx', degrees=True)[0])
                # Error de guiñada relativo [-pi, pi]
                yaw_error_rad = np.radians((yaw_rel_deg + 180) % 360 - 180)

                rot = R_scipy.from_matrix(rmat)
                euler_deg = rot.as_euler('xyz', degrees=True)
            else:
                yaw_error_rad = 0.0
        else:
            yaw_error_rad = 0.0

        # 3. Determinar si Kalman sigue activo por predicción ("coasting")
        time_since_detection = now - self.last_detection_time
        kalman_active = self.kf.is_initialized and (time_since_detection < self.max_coast_time_s)

        filtered_pos_body = self.kf.position if kalman_active else None
        filtered_vel_body = self.kf.velocity if kalman_active else None

        # 4. Proyección a marco World NED
        if filtered_pos_body is not None and current_attitude_rad is not None:
            yaw = current_attitude_rad[2]
            cos_y, sin_y = np.cos(yaw), np.sin(yaw)
            R_yaw_inv = np.array([
                [cos_y, -sin_y, 0.0],
                [sin_y, cos_y, 0.0],
                [0.0, 0.0, 1.0]
            ])
            tvec_world = R_yaw_inv @ filtered_pos_body

        return {
            "detected": detected,
            "marker_id": target_id,
            "corners": corners,
            "rvec": rvec,
            "tvec_cam": tvec_cam,
            "tvec_body_raw": tvec_body,
            "pos_body_filtered": filtered_pos_body,
            "vel_body_filtered": filtered_vel_body,
            "pos_world_rel": tvec_world,
            "euler_deg": euler_deg,
            "yaw_error_rad": yaw_error_rad,
            "kalman_active": kalman_active,
            "time_since_detect": time_since_detection
        }
