import cv2
import numpy as np
import json
import base64
import math
import time
import threading
import asyncio
import websockets
from http.server import SimpleHTTPRequestHandler
import socketserver
import csv
from datetime import datetime
from scipy.spatial.transform import Rotation as R_scipy

class FlightTrackerLogger:
    def __init__(self):
        self.file = None
        self.writer = None
        self.is_logging = False
        self.lock = threading.Lock()

    def start_session(self):
        with self.lock:
            if self.is_logging:
                return
            filename = f"aruco_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            self.file = open(filename, mode='w', newline='')
            self.writer = csv.writer(self.file)
            
            # Encabezados incluyendo cuaterniones y euler
            self.writer.writerow([
                "timestamp", "frame_id", "detected", "tracked_marker", "kalman_active",
                "raw_x", "raw_y", "raw_z", 
                "kalman_x", "kalman_y", "kalman_z",
                "qw", "qx", "qy", "qz",
                "pitch", "yaw", "roll"
            ])
            self.is_logging = True
            print(f"[LOGGING] Logging iniciado: {filename}")

    def log_frame(self, frame_id, detected, marker_id, kalman_active, raw_pos, kalman_pos, quat, euler_angles):
        with self.lock:
            if not self.is_logging or self.writer is None:
                return
                
            rx, ry, rz = raw_pos
            kx, ky, kz = kalman_pos
            qw, qx, qy, qz = quat
            pitch, yaw, roll = euler_angles

            self.writer.writerow([
                time.time(), frame_id, detected, marker_id, kalman_active,
                f"{rx:.4f}", f"{ry:.4f}", f"{rz:.4f}",
                f"{kx:.4f}", f"{ky:.4f}", f"{kz:.4f}",
                f"{qw:.4f}", f"{qx:.4f}", f"{qy:.4f}", f"{qz:.4f}",
                f"{pitch:.2f}", f"{yaw:.2f}", f"{roll:.2f}"
            ])

    def stop_session(self):
        with self.lock:
            if not self.is_logging:
                return
            if self.file:
                self.file.close()
                self.file = None
                self.writer = None
            self.is_logging = False
            print("[LOGGING] Logging finalizado y guardado.")

logger = FlightTrackerLogger()
frame_count = 0

# Global state to share between OpenCV thread and WebSocket thread
latest_data = {
    "detected": False,
    "x": 0.0,
    "y": 0.0,
    "z": 0.0,
    "pitch": 0.0,
    "yaw": 0.0,
    "roll": 0.0,
    "rotation_matrix": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    "tracked_marker": -1,
    "kalman_active": False,
    "frame": "",
    "is_logging": False
}
data_lock = threading.Lock()

marker_size = 0.075
target_marker_id = 0

# Kalman Filter 1D class
class Kalman1D:
    def __init__(self, process_noise=0.05, measurement_noise=0.15, error_covariance=1.0):
        self.x = 0.0  # state (position / quat component)
        self.v = 0.0  # state (velocity)
        self.P = np.array([[error_covariance, 0.0],
                           [0.0, error_covariance]], dtype=np.float32)
        self.Q = np.array([[process_noise, 0.0],
                           [0.0, process_noise]], dtype=np.float32)
        self.R = measurement_noise

    def predict(self, dt):
        F = np.array([[1.0, dt],
                      [0.0, 1.0]], dtype=np.float32)
        self.x = float(self.x + self.v * dt)
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        S = self.P[0, 0] + self.R
        K = np.array([self.P[0, 0] / S,
                      self.P[1, 0] / S], dtype=np.float32)
        innovation = z - self.x
        self.x = float(self.x + K[0] * innovation)
        self.v = float(self.v + K[1] * innovation)
        I_KH = np.array([[1.0 - K[0], 0.0],
                         [-K[1], 1.0]], dtype=np.float32)
        self.P = I_KH @ self.P

    def reset(self, initial_position):
        self.x = float(initial_position)
        self.v = 0.0
        self.P = np.eye(2, dtype=np.float32) * 1.0

# 3 Kalman Filters for Position (x, y, z)
kf_x = Kalman1D()
kf_y = Kalman1D()
kf_z = Kalman1D()

# 4 Kalman Filters for Quaternion Orientation (qw, qx, qy, qz)
kf_qw = Kalman1D()
kf_qx = Kalman1D()
kf_qy = Kalman1D()
kf_qz = Kalman1D()

kalman_predict_count = 0
MAX_KALMAN_PREDICT_FRAMES = 15
last_measurement_time = None

def reset_kalman(x, y, z, qw, qx, qy, qz):
    kf_x.reset(x)
    kf_y.reset(y)
    kf_z.reset(z)
    kf_qw.reset(qw)
    kf_qx.reset(qx)
    kf_qy.reset(qy)
    kf_qz.reset(qz)

def update_kalman_params(q, r):
    for kf in [kf_x, kf_y, kf_z, kf_qw, kf_qx, kf_qy, kf_qz]:
        kf.Q = np.array([[q, 0.0], [0.0, q]], dtype=np.float32)
        kf.R = r

focal_length_factor = 1.6

brightness = 0.0        
contrast = 1.0          
auto_contrast = True    
show_processed = False  
clahe_clip_limit = 3.0  
clahe_grid_size = 8     
noise_reduction = 0     

roi_tracking_enabled = False
bilateral_filtering = True
bilateral_d = 5
bilateral_sigma_color = 75
bilateral_sigma_space = 75
morphology_enabled = True
morphology_kernel_size = 3
kalman_enabled = True
kalman_q = 0.05
kalman_r = 0.15
tracking_mode = "dual"

roi_active = False
roi_box = None  

min_marker_perimeter_rate = 0.005
poly_approx_accuracy_rate = 0.055
max_erroneous_bits_border = 0.5
error_correction_rate = 0.8

cap = None
current_camera_index = 0
requested_camera_index = 0
camera_changed = False
is_switching_camera = False
available_cameras = []

def scan_available_cameras():
    global available_cameras
    available_cameras = []
    print("Camera Scanner: Scanning indices 0 to 4...")
    for idx in range(5):
        try:
            temp_cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if temp_cap is None or not temp_cap.isOpened():
                temp_cap = cv2.VideoCapture(idx)
                
            if temp_cap is not None and temp_cap.isOpened():
                ret, _ = temp_cap.read()
                if ret:
                    available_cameras.append(idx)
                temp_cap.release()
        except Exception as e:
            print(f"Camera Scanner error on index {idx}: {e}")
    
    print(f"Camera Scanner: Found active cameras: {available_cameras}")
    if not available_cameras:
        available_cameras = [0]

dict_type = cv2.aruco.DICT_4X4_50
if hasattr(cv2.aruco, 'getPredefinedDictionary'):
    dictionary = cv2.aruco.getPredefinedDictionary(dict_type)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.minMarkerPerimeterRate = 0.015
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 23
    parameters.adaptiveThreshWinSizeStep = 4
    parameters.polygonalApproxAccuracyRate = 0.055
    parameters.maxErroneousBitsInBorderRate = 0.5
    parameters.errorCorrectionRate = 0.8
    
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    def detect_markers(image):
        return detector.detectMarkers(image)
else:
    dictionary = cv2.aruco.Dictionary_get(dict_type)
    parameters = cv2.aruco.DetectorParameters_create()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.minMarkerPerimeterRate = 0.015
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 23
    parameters.adaptiveThreshWinSizeStep = 4
    parameters.polygonalApproxAccuracyRate = 0.055
    parameters.maxErroneousBitsInBorderRate = 0.5
    parameters.errorCorrectionRate = 0.8
    
    def detect_markers(image):
        return cv2.aruco.detectMarkers(image, dictionary, parameters=parameters)

def draw_axes(img, K, dist, rvec, tvec, length):
    try:
        if hasattr(cv2, 'drawFrameAxes'):
            cv2.drawFrameAxes(img, K, dist, rvec, tvec, length)
        elif hasattr(cv2.aruco, 'drawAxis'):
            cv2.aruco.drawAxis(img, K, dist, rvec, tvec, length)
    except Exception as e:
        print(f"Error drawing axis: {e}")

# Convierte Matriz de Rotación 3x3 a Cuaternión [qw, qx, qy, qz] de forma robusta
def matrix_to_quaternion(R_mat):
    try:
        if not np.all(np.isfinite(R_mat)):
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # Algoritmo de Shepperd para máxima estabilidad numérica
        t = float(np.trace(R_mat))
        if t > 0.0:
            s = math.sqrt(t + 1.0) * 2.0
            qw = 0.25 * s
            qx = float((R_mat[2, 1] - R_mat[1, 2]) / s)
            qy = float((R_mat[0, 2] - R_mat[2, 0]) / s)
            qz = float((R_mat[1, 0] - R_mat[0, 1]) / s)
        elif (R_mat[0, 0] > R_mat[1, 1]) and (R_mat[0, 0] > R_mat[2, 2]):
            s = math.sqrt(max(1.0 + float(R_mat[0, 0] - R_mat[1, 1] - R_mat[2, 2]), 1e-6)) * 2.0
            qw = float((R_mat[2, 1] - R_mat[1, 2]) / s)
            qx = 0.25 * s
            qy = float((R_mat[0, 1] + R_mat[1, 0]) / s)
            qz = float((R_mat[0, 2] + R_mat[2, 0]) / s)
        elif R_mat[1, 1] > R_mat[2, 2]:
            s = math.sqrt(max(1.0 + float(R_mat[1, 1] - R_mat[0, 0] - R_mat[2, 2]), 1e-6)) * 2.0
            qw = float((R_mat[0, 2] - R_mat[2, 0]) / s)
            qx = float((R_mat[0, 1] + R_mat[1, 0]) / s)
            qy = 0.25 * s
            qz = float((R_mat[1, 2] + R_mat[2, 1]) / s)
        else:
            s = math.sqrt(max(1.0 + float(R_mat[2, 2] - R_mat[0, 0] - R_mat[1, 1]), 1e-6)) * 2.0
            qw = float((R_mat[1, 0] - R_mat[0, 1]) / s)
            qx = float((R_mat[0, 2] + R_mat[2, 0]) / s)
            qy = float((R_mat[1, 2] + R_mat[2, 1]) / s)
            qz = 0.25 * s

        q = np.array([qw, qx, qy, qz], dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm > 1e-6:
            q = q / norm
        else:
            q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return q
    except Exception:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

# Convierte Cuaternión [qw, qx, qy, qz] a Matriz 3x3 y Ángulos Euler (radians)
def quaternion_to_matrix_and_euler(q):
    try:
        norm = np.linalg.norm(q)
        if norm < 1e-6 or not np.all(np.isfinite(q)):
            q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            q = q / norm
            
        qw, qx, qy, qz = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        
        R_mat = np.array([
            [1.0 - 2.0*(qy*qy + qz*qz), 2.0*(qx*qy - qz*qw), 2.0*(qx*qz + qy*qw)],
            [2.0*(qx*qy + qz*qw), 1.0 - 2.0*(qx*qx + qz*qz), 2.0*(qy*qz - qx*qw)],
            [2.0*(qx*qz - qy*qw), 2.0*(qy*qz + qx*qw), 1.0 - 2.0*(qx*qx + qy*qy)]
        ], dtype=np.float32)
        
        sy = math.sqrt(max(float(R_mat[0, 0] * R_mat[0, 0] + R_mat[1, 0] * R_mat[1, 0]), 0.0))
        singular = sy < 1e-6
        if not singular:
            x = math.atan2(float(R_mat[2, 1]), float(R_mat[2, 2]))
            y = math.atan2(-float(R_mat[2, 0]), sy)
            z = math.atan2(float(R_mat[1, 0]), float(R_mat[0, 0]))
        else:
            x = math.atan2(-float(R_mat[1, 2]), float(R_mat[1, 1]))
            y = math.atan2(-float(R_mat[2, 0]), sy)
            z = 0.0
            
        return R_mat, (x, y, z), q
    except Exception:
        return np.eye(3, dtype=np.float32), (0.0, 0.0, 0.0), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

def run_http_server():
    PORT = 8000
    class QuietHandler(SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
            super().end_headers()

        def log_message(self, format, *args):
            pass

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), QuietHandler) as httpd:
        print(f"HTTP Server: Running on http://localhost:{PORT}")
        httpd.serve_forever()

async def ws_handler(websocket):
    print(f"WebSocket Client Connected: {websocket.remote_address}")
    try:
        await websocket.send(json.dumps({
            "type": "camera_list",
            "cameras": available_cameras
        }))
    except Exception as e:
        print(f"Error sending camera list: {e}")
    
    async def send_loop():
        try:
            while True:
                with data_lock:
                    message = json.dumps(latest_data)
                await websocket.send(message)
                await asyncio.sleep(1.0 / 30.0)
        except asyncio.CancelledError:
            pass
            
    async def recv_loop():
        global marker_size, focal_length_factor, brightness, contrast, auto_contrast, show_processed, clahe_clip_limit, clahe_grid_size, min_marker_perimeter_rate, poly_approx_accuracy_rate, max_erroneous_bits_border, error_correction_rate, noise_reduction, requested_camera_index, camera_changed
        global roi_tracking_enabled, bilateral_filtering, morphology_enabled, morphology_kernel_size, kalman_enabled, kalman_q, kalman_r, tracking_mode
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if data.get("type") == "calibrate":
                        marker_size = float(data["marker_size"])
                        focal_length_factor = float(data["focal_length_factor"])
                        print(f"Calibration updated: marker_size={marker_size}m, focal_length={focal_length_factor}")
                    elif data.get("type") == "settings":
                        brightness = float(data["brightness"])
                        contrast = float(data["contrast"])
                        auto_contrast = bool(data["auto_contrast"])
                        show_processed = bool(data["show_processed"])
                        clahe_clip_limit = float(data["clahe_clip_limit"])
                        clahe_grid_size = int(data["clahe_grid_size"])
                        noise_reduction = int(data.get("noise_reduction", 0))
                        
                        min_marker_perimeter_rate = float(data["min_marker_perimeter_rate"])
                        poly_approx_accuracy_rate = float(data["poly_approx_accuracy_rate"])
                        max_erroneous_bits_border = float(data["max_erroneous_bits_border"])
                        error_correction_rate = float(data["error_correction_rate"])
                        
                        roi_tracking_enabled = bool(data.get("roi_tracking_enabled", roi_tracking_enabled))
                        bilateral_filtering = bool(data.get("bilateral_filtering", bilateral_filtering))
                        morphology_enabled = bool(data.get("morphology_enabled", morphology_enabled))
                        morphology_kernel_size = int(data.get("morphology_kernel_size", morphology_kernel_size))
                        kalman_enabled = bool(data.get("kalman_enabled", kalman_enabled))
                        kalman_q = float(data.get("kalman_q", kalman_q))
                        kalman_r = float(data.get("kalman_r", kalman_r))
                        tracking_mode = str(data.get("tracking_mode", tracking_mode))
                        
                        update_kalman_params(kalman_q, kalman_r)
                    elif data.get("type") == "change_camera":
                        requested_camera_index = int(data["index"])
                        camera_changed = True
                    elif data.get("type") == "start_logging":
                        logger.start_session()
                    elif data.get("type") == "stop_logging":
                        logger.stop_session()
                except Exception as e:
                    print(f"Error parsing client settings: {e}")
        except asyncio.CancelledError:
            pass
            
    send_task = asyncio.create_task(send_loop())
    recv_task = asyncio.create_task(recv_loop())
    
    try:
        await asyncio.gather(send_task, recv_task)
    except websockets.exceptions.ConnectionClosed:
        print(f"WebSocket Client Disconnected: {websocket.remote_address}")
    finally:
        send_task.cancel()
        recv_task.cancel()

def run_ws_server():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    async def start():
        async with websockets.serve(ws_handler, "localhost", 8765):
            print("WebSocket Server: Running on ws://localhost:8765")
            await asyncio.Future()
            
    loop.run_until_complete(start())

def async_open_camera(index):
    global cap, current_camera_index, is_switching_camera
    print(f"Async Camera Opener: Initiating thread to open camera {index}...")
    try:
        temp_cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if temp_cap is None or not temp_cap.isOpened():
            temp_cap = cv2.VideoCapture(index)
            
        if temp_cap is not None and temp_cap.isOpened():
            temp_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            temp_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ret, _ = temp_cap.read()
            if ret:
                old_cap = cap
                cap = temp_cap
                current_camera_index = index
                if old_cap is not None:
                    try:
                        old_cap.release()
                    except Exception:
                        pass
                print(f"Async Camera Opener: Successfully switched to camera {current_camera_index}")
            else:
                temp_cap.release()
        else:
            if temp_cap is not None:
                temp_cap.release()
    except Exception as e:
        print(f"Async Camera Opener EXCEPTION: {e}")
    finally:
        is_switching_camera = False

def run_camera_tracking():
    global latest_data, marker_size, focal_length_factor, brightness, contrast, auto_contrast, show_processed, clahe_clip_limit, clahe_grid_size, noise_reduction, min_marker_perimeter_rate, poly_approx_accuracy_rate, max_erroneous_bits_border, error_correction_rate, current_camera_index, requested_camera_index, camera_changed, is_switching_camera, cap
    global roi_tracking_enabled, bilateral_filtering, bilateral_d, bilateral_sigma_color, bilateral_sigma_space, morphology_enabled, morphology_kernel_size, kalman_enabled, kalman_q, kalman_r, roi_active, roi_box, kalman_predict_count, last_measurement_time, tracking_mode
    global frame_count
    
    cap = None
    for idx in [0, 1, 2]:
        temp_cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if temp_cap is None or not temp_cap.isOpened():
            temp_cap = cv2.VideoCapture(idx)
            
        if temp_cap is not None and temp_cap.isOpened():
            cap = temp_cap
            current_camera_index = idx
            requested_camera_index = idx
            print(f"Camera Tracker: Initialized webcam on index {idx}")
            break
        if temp_cap is not None:
            temp_cap.release()

    if cap is None:
        print("Camera Tracker WARNING: No webcam could be opened!")
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    width = 640
    height = 480
    print("Camera Tracker: Tracker loop started.")
    
    while True:
        loop_start = time.time()
        frame_count += 1
        
        if camera_changed:
            camera_changed = False
            if not is_switching_camera:
                is_switching_camera = True
                thread = threading.Thread(target=async_open_camera, args=(requested_camera_index,), daemon=True)
                thread.start()
        
        if is_switching_camera or cap is None:
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(dummy_frame, f"Cargando Camara {requested_camera_index}...", (160, 220), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 180, 50), 2, cv2.LINE_AA)
            
            _, buffer = cv2.imencode('.jpg', dummy_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            with data_lock:
                latest_data = {
                    "detected": False, "x": 0.0, "y": 0.0, "z": 0.0,
                    "pitch": 0.0, "yaw": 0.0, "roll": 0.0,
                    "rotation_matrix": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    "frame": frame_base64,
                    "is_logging": logger.is_logging
                }
            time.sleep(0.1)
            continue
            
        ret = False
        frame = None
        if cap is not None:
            try:
                ret, frame = cap.read()
            except Exception as e:
                ret = False
            
        if ret and frame is not None:
            height, width = frame.shape[:2]
            f_px = width * focal_length_factor
            K = np.array([[f_px, 0.0, width / 2.0], [0.0, f_px, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
            dist_coeffs = np.zeros((4, 1), dtype=np.float32)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if bilateral_filtering:
                gray = cv2.bilateralFilter(gray, bilateral_d, bilateral_sigma_color, bilateral_sigma_space)
            
            gray = cv2.convertScaleAbs(gray, alpha=contrast, beta=brightness)
            if auto_contrast:
                grid = max(2, clahe_grid_size)
                clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(grid, grid))
                gray = clahe.apply(gray)
            
            if noise_reduction > 0:
                ksize = int(noise_reduction)
                if ksize % 2 == 0: ksize += 1
                gray = cv2.medianBlur(gray, ksize)
            
            if morphology_enabled:
                k_size = int(morphology_kernel_size)
                if k_size % 2 == 0: k_size += 1
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
                gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
                gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            
            if hasattr(cv2.aruco, 'getPredefinedDictionary'):
                params = detector.getDetectorParameters()
                params.minMarkerPerimeterRate = min_marker_perimeter_rate
                params.polygonalApproxAccuracyRate = poly_approx_accuracy_rate
                params.maxErroneousBitsInBorderRate = max_erroneous_bits_border
                params.errorCorrectionRate = error_correction_rate
                detector.setDetectorParameters(params)
            else:
                parameters.minMarkerPerimeterRate = min_marker_perimeter_rate
                parameters.polygonalApproxAccuracyRate = poly_approx_accuracy_rate
                parameters.maxErroneousBitsInBorderRate = max_erroneous_bits_border
                parameters.errorCorrectionRate = error_correction_rate
                
            if show_processed:
                if hasattr(cv2.aruco, 'getPredefinedDictionary'):
                    block_size = int(detector.getDetectorParameters().adaptiveThreshWinSizeMax)
                else:
                    block_size = int(parameters.adaptiveThreshWinSizeMax)
                if block_size % 2 == 0: block_size += 1
                block_size = max(3, block_size)
                
                thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, 7)
                display_frame = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            else:
                display_frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
            
            corners, ids, rejected = [], None, []
            roi_used_this_frame = False
            
            if roi_tracking_enabled and roi_active and roi_box is not None:
                rx1, ry1, rx2, ry2 = roi_box
                if (rx2 - rx1) > 40 and (ry2 - ry1) > 40:
                    gray_roi = gray[ry1:ry2, rx1:rx2]
                    corners_roi, ids_roi, rejected_roi = detect_markers(gray_roi)
                    if ids_roi is not None and len(ids_roi) > 0:
                        corners = []
                        for c in corners_roi:
                            c_shifted = c.copy()
                            c_shifted[0] += np.array([rx1, ry1], dtype=np.float32)
                            corners.append(c_shifted)
                        ids = ids_roi
                        roi_used_this_frame = True
            
            if not roi_used_this_frame:
                corners, ids, rejected = detect_markers(gray)
                roi_active = False
            
            detected = False
            tx = ty = tz = 0.0
            raw_tx = raw_ty = raw_tz = 0.0
            pitch = yaw = roll = 0.0
            quat_out = [1.0, 0.0, 0.0, 0.0]
            R_flat = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            tracked_marker = -1
            kalman_active = False
            
            if ids is not None and len(ids) > 0:
                ids_flat = ids.flatten()
                idx = -1
                size_phys = marker_size
                offset_local = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)
                
                if tracking_mode == "single_small":
                    for i, mid in enumerate(ids_flat):
                        if mid == 1:
                            idx, tracked_marker = i, 1
                            break
                elif tracking_mode == "single_large":
                    for i, mid in enumerate(ids_flat):
                        if mid == 0:
                            idx, tracked_marker = i, 0
                            break
                else:  # "dual"
                    for i, mid in enumerate(ids_flat):
                        if mid == 1:
                            idx, tracked_marker = i, 1
                            size_phys = marker_size / 5.0
                            break
                    if idx == -1:
                        for i, mid in enumerate(ids_flat):
                            if mid == 0:
                                idx, tracked_marker = i, 0
                                size_phys = marker_size
                                offset_local = np.array([[0.0], [marker_size * (230.0 / 300.0)], [0.0]], dtype=np.float32)
                                break
                
                if idx != -1:
                    img_points = corners[idx][0].astype(np.float32)
                    obj_points = np.array([
                        [-size_phys/2.0, size_phys/2.0, 0.0],
                        [size_phys/2.0, size_phys/2.0, 0.0],
                        [size_phys/2.0, -size_phys/2.0, 0.0],
                        [-size_phys/2.0, -size_phys/2.0, 0.0]
                    ], dtype=np.float32)
                    
                    try:
                        success, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                    except Exception:
                        success, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, dist_coeffs)
                        
                    if success:
                        detected = True
                        R, _ = cv2.Rodrigues(rvec)
                        tvec_landing = tvec + R @ offset_local
                        
                        tx = float(tvec_landing[0][0])
                        ty = float(tvec_landing[1][0])
                        tz = float(tvec_landing[2][0])
                        raw_tx, raw_ty, raw_tz = tx, ty, tz
                        
                        # Convertir matriz R de OpenCV a Cuaternión [qw, qx, qy, qz]
                        q_raw = matrix_to_quaternion(R)
                        qw, qx, qy, qz = q_raw
                        
                        if kalman_enabled:
                            now = time.time()
                            dt = 1.0 / 30.0 if last_measurement_time is None else (now - last_measurement_time)
                            last_measurement_time = now
                            
                            # Alineación de signo para evitar "quaternion hemisphere flip" (+q y -q son la misma rotación)
                            if kalman_predict_count == 0:
                                current_q_state = np.array([kf_qw.x, kf_qx.x, kf_qy.x, kf_qz.x])
                                if np.dot(current_q_state, q_raw) < 0:
                                    qw, qx, qy, qz = -qw, -qx, -qy, -qz
                            
                            if kalman_predict_count > 0:
                                reset_kalman(tx, ty, tz, qw, qx, qy, qz)
                                kalman_predict_count = 0
                            else:
                                kf_x.predict(dt); kf_y.predict(dt); kf_z.predict(dt)
                                kf_qw.predict(dt); kf_qx.predict(dt); kf_qy.predict(dt); kf_qz.predict(dt)
                                
                                kf_x.update(tx); kf_y.update(ty); kf_z.update(tz)
                                kf_qw.update(qw); kf_qx.update(qx); kf_qy.update(qy); kf_qz.update(qz)
                                
                            tx, ty, tz = kf_x.x, kf_y.x, kf_z.x
                            q_filt = np.array([kf_qw.x, kf_qx.x, kf_qy.x, kf_qz.x], dtype=np.float32)
                        else:
                            q_filt = q_raw
                            
                        # Reconstruir Matriz R y ángulos de Euler desde el cuaternión filtrado
                        R_smooth, (pitch, yaw, roll), quat_out = quaternion_to_matrix_and_euler(q_filt)
                        
                        rvec_smooth, _ = cv2.Rodrigues(R_smooth)
                        tvec_drawn = tvec_landing - R_smooth @ offset_local
                            
                        R_flat = R_smooth.flatten().tolist()
                        
                        if roi_tracking_enabled:
                            c_pts = corners[idx][0]
                            x_min, x_max = np.min(c_pts[:, 0]), np.max(c_pts[:, 0])
                            y_min, y_max = np.min(c_pts[:, 1]), np.max(c_pts[:, 1])
                            w, h = x_max - x_min, y_max - y_min
                            cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
                            roi_box = [max(0, int(cx - w * 1.5)), max(0, int(cy - h * 1.5)),
                                       min(width, int(cx + w * 1.5)), min(height, int(cy + h * 1.5))]
                            roi_active = True
                        
                        cv2.aruco.drawDetectedMarkers(display_frame, [corners[idx]], np.array([[tracked_marker]]))
                        draw_axes(display_frame, K, dist_coeffs, rvec_smooth, tvec_drawn, size_phys * 0.5)
                        if roi_tracking_enabled and roi_active:
                            cv2.rectangle(display_frame, (roi_box[0], roi_box[1]), (roi_box[2], roi_box[3]), (0, 255, 0), 2)
            
            if not detected:
                if kalman_enabled and last_measurement_time is not None and kalman_predict_count < MAX_KALMAN_PREDICT_FRAMES:
                    now = time.time()
                    dt = now - last_measurement_time
                    last_measurement_time = now
                    kalman_predict_count += 1
                    
                    kf_x.predict(dt); kf_y.predict(dt); kf_z.predict(dt)
                    kf_qw.predict(dt); kf_qx.predict(dt); kf_qy.predict(dt); kf_qz.predict(dt)
                    
                    tx, ty, tz = kf_x.x, kf_y.x, kf_z.x
                    q_filt = np.array([kf_qw.x, kf_qx.x, kf_qy.x, kf_qz.x], dtype=np.float32)
                    
                    R_smooth, (pitch, yaw, roll), quat_out = quaternion_to_matrix_and_euler(q_filt)
                    R_flat = R_smooth.flatten().tolist()
                    
                    detected = True
                    kalman_active = True
                    cv2.putText(display_frame, "KALMAN ESTIMATING...", (15, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)
                else:
                    roi_active = False
                    if kalman_predict_count >= MAX_KALMAN_PREDICT_FRAMES:
                        last_measurement_time = None

            # Registro CSV extendido con Posición, Cuaterniones y Euler
            logger.log_frame(
                frame_id=frame_count,
                detected=detected,
                marker_id=tracked_marker,
                kalman_active=kalman_active,
                raw_pos=(raw_tx, raw_ty, raw_tz),
                kalman_pos=(tx, ty, tz),
                quat=quat_out,
                euler_angles=(pitch, yaw, roll)
            )

            _, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            with data_lock:
                latest_data = {
                    "detected": detected, "x": tx, "y": ty, "z": tz,
                    "pitch": pitch, "yaw": yaw, "roll": roll,
                    "rotation_matrix": R_flat, "tracked_marker": tracked_marker,
                    "kalman_active": kalman_active, "frame": frame_base64,
                    "is_logging": logger.is_logging
                }
        else:
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(dummy_frame, "Webcam Offline / In Use", (140, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 220), 2, cv2.LINE_AA)
            _, buffer = cv2.imencode('.jpg', dummy_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            with data_lock:
                latest_data = {
                    "detected": False, "x": 0.0, "y": 0.0, "z": 0.0,
                    "pitch": 0.0, "yaw": 0.0, "roll": 0.0,
                    "rotation_matrix": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    "frame": frame_base64,
                    "is_logging": logger.is_logging
                }
        
        elapsed = time.time() - loop_start
        sleep_time = max(1.0/30.0 - elapsed, 0)
        time.sleep(sleep_time)

if __name__ == "__main__":
    scan_available_cameras()
    
    http_thread = threading.Thread(target=run_http_server, daemon=True)
    http_thread.start()
    
    ws_thread = threading.Thread(target=run_ws_server, daemon=True)
    ws_thread.start()
    
    try:
        run_camera_tracking()
    except KeyboardInterrupt:
        print("\nStopping servers...")