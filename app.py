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
    "frame": ""
}
data_lock = threading.Lock()

# Define the physical size of the ArUco marker (in meters)
# Default is 5cm (0.05m). The user can adjust this via calibration.
marker_size = 0.05
target_marker_id = 0

# Kalman Filter 1D class
class Kalman1D:
    def __init__(self, process_noise=0.05, measurement_noise=0.15, error_covariance=1.0):
        self.x = 0.0  # state (position)
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

# Initialize 6 Kalman Filters for pose tracking (x, y, z, rx, ry, rz)
kf_x = Kalman1D()
kf_y = Kalman1D()
kf_z = Kalman1D()
kf_rx = Kalman1D()
kf_ry = Kalman1D()
kf_rz = Kalman1D()

# Track how many frames we've predicted without measurements
kalman_predict_count = 0
MAX_KALMAN_PREDICT_FRAMES = 15
last_measurement_time = None

def reset_kalman(x, y, z, rx, ry, rz):
    kf_x.reset(x)
    kf_y.reset(y)
    kf_z.reset(z)
    kf_rx.reset(rx)
    kf_ry.reset(ry)
    kf_rz.reset(rz)

def update_kalman_params(q, r):
    for kf in [kf_x, kf_y, kf_z, kf_rx, kf_ry, kf_rz]:
        kf.Q = np.array([[q, 0.0], [0.0, q]], dtype=np.float32)
        kf.R = r

# Camera focal length scaling factor (approximation)
focal_length_factor = 0.8

# Video preprocessing and advanced parameters
brightness = 0.0        # -100 to 100
contrast = 1.0          # 0.5 to 3.0
auto_contrast = True    # Use CLAHE
show_processed = False  # If true, stream preprocessed/grayscale/thresholded frame to client
clahe_clip_limit = 3.0  # CLAHE clip limit
clahe_grid_size = 8     # CLAHE tile grid size (8x8)
noise_reduction = 0     # 0 = Off, 3 = 3x3 median blur, etc.

# New robust parameters
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
tracking_mode = "dual" # "dual", "single_small", "single_large"

# ROI tracking state
roi_active = False
roi_box = None  # [x1, y1, x2, y2]

# ArUco detection parameters
min_marker_perimeter_rate = 0.015
poly_approx_accuracy_rate = 0.055
max_erroneous_bits_border = 0.5
error_correction_rate = 0.8

# Camera source state
cap = None
current_camera_index = 0
requested_camera_index = 0
camera_changed = False
is_switching_camera = False
available_cameras = []

# Scan available cameras at startup
def scan_available_cameras():
    global available_cameras
    available_cameras = []
    print("Camera Scanner: Scanning indices 0 to 4...")
    for idx in range(5):
        try:
            # Try DirectShow first on Windows as it is faster and doesn't block
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
        available_cameras = [0]  # Fallback to index 0

# Set up ArUco Detector with optimized parameters for phone screens and small/far/curved markers
dict_type = cv2.aruco.DICT_4X4_50
if hasattr(cv2.aruco, 'getPredefinedDictionary'):
    dictionary = cv2.aruco.getPredefinedDictionary(dict_type)
    parameters = cv2.aruco.DetectorParameters()
    
    # Enable corner refinement for high accuracy and stability
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    # Allow smaller markers to be detected (further away)
    parameters.minMarkerPerimeterRate = 0.015
    # Customize thresholding for handling screen reflections
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 23
    parameters.adaptiveThreshWinSizeStep = 4
    # Relax polygon approximation to allow curved/deformed edges
    parameters.polygonalApproxAccuracyRate = 0.055
    # Be more tolerant to errors in the black border
    parameters.maxErroneousBitsInBorderRate = 0.5
    # Increase error correction capability
    parameters.errorCorrectionRate = 0.8
    
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    def detect_markers(image):
        return detector.detectMarkers(image)
else:
    dictionary = cv2.aruco.Dictionary_get(dict_type)
    parameters = cv2.aruco.DetectorParameters_create()
    
    # Enable corner refinement
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.minMarkerPerimeterRate = 0.015
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 23
    parameters.adaptiveThreshWinSizeStep = 4
    # Relax polygon approximation
    parameters.polygonalApproxAccuracyRate = 0.055
    parameters.maxErroneousBitsInBorderRate = 0.5
    parameters.errorCorrectionRate = 0.8
    
    def detect_markers(image):
        return cv2.aruco.detectMarkers(image, dictionary, parameters=parameters)


# Safe function to draw axis
def draw_axes(img, K, dist, rvec, tvec, length):
    try:
        if hasattr(cv2, 'drawFrameAxes'):
            cv2.drawFrameAxes(img, K, dist, rvec, tvec, length)
        elif hasattr(cv2.aruco, 'drawAxis'):
            cv2.aruco.drawAxis(img, K, dist, rvec, tvec, length)
    except Exception as e:
        print(f"Error drawing axis: {e}")

# Helper to convert rotation matrix to Euler angles
def get_euler_angles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return x, y, z # pitch, yaw, roll (radians)

# HTTP static file server thread
def run_http_server():
    PORT = 8000
    class QuietHandler(SimpleHTTPRequestHandler):
        # Override log_message to prevent console cluttering from assets loading
        def log_message(self, format, *args):
            pass

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), QuietHandler) as httpd:
        print(f"HTTP Server: Running on http://localhost:{PORT}")
        httpd.serve_forever()

# WebSocket server handler
async def ws_handler(websocket):
    print(f"WebSocket Client Connected: {websocket.remote_address}")
    
    # Send the list of available cameras right away
    try:
        await websocket.send(json.dumps({
            "type": "camera_list",
            "cameras": available_cameras
        }))
    except Exception as e:
        print(f"Error sending camera list: {e}")
    
    # Task to send tracking data to client at 30 FPS
    async def send_loop():
        try:
            while True:
                with data_lock:
                    message = json.dumps(latest_data)
                await websocket.send(message)
                await asyncio.sleep(1.0 / 30.0)
        except asyncio.CancelledError:
            pass
            
    # Task to receive calibration settings from client
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
                        print(f"Calibration updated by client: marker_size={marker_size}m, focal_length_factor={focal_length_factor}")
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
                        
                        print(f"Settings updated: brightness={brightness}, contrast={contrast}, auto_contrast={auto_contrast}, show_processed={show_processed}, min_perimeter={min_marker_perimeter_rate}, poly_approx={poly_approx_accuracy_rate}, max_border_err={max_erroneous_bits_border}, err_correction={error_correction_rate}, roi_enabled={roi_tracking_enabled}, bilateral={bilateral_filtering}, morphology={morphology_enabled}, kalman={kalman_enabled}, tracking_mode={tracking_mode}")
                    elif data.get("type") == "change_camera":
                        requested_camera_index = int(data["index"])
                        camera_changed = True
                        print(f"Camera change requested: index={requested_camera_index}")
                except Exception as e:
                    print(f"Error parsing client settings: {e}")
        except asyncio.CancelledError:
            pass
            
    # Run both loops concurrently
    send_task = asyncio.create_task(send_loop())
    recv_task = asyncio.create_task(recv_loop())
    
    try:
        await asyncio.gather(send_task, recv_task)
    except websockets.exceptions.ConnectionClosed:
        print(f"WebSocket Client Disconnected: {websocket.remote_address}")
    finally:
        send_task.cancel()
        recv_task.cancel()


# WebSocket server thread
def run_ws_server():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    async def start():
        async with websockets.serve(ws_handler, "localhost", 8765):
            print("WebSocket Server: Running on ws://localhost:8765")
            await asyncio.Future() # Keep running
            
    loop.run_until_complete(start())

# Helper to open camera in a separate thread to prevent blocking the main process on Windows
def async_open_camera(index):
    global cap, current_camera_index, is_switching_camera
    print(f"Async Camera Opener: Initiating thread to open camera {index}...")
    try:
        # Try DirectShow first on Windows as it is faster and doesn't block
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
                print(f"Async Camera Opener ERROR: Could not read frame from camera {index}")
        else:
            if temp_cap is not None:
                temp_cap.release()
            print(f"Async Camera Opener ERROR: Could not open camera {index}")
    except Exception as e:
        print(f"Async Camera Opener EXCEPTION: {e}")
    finally:
        is_switching_camera = False

def run_camera_tracking():
    global latest_data, marker_size, focal_length_factor, brightness, contrast, auto_contrast, show_processed, clahe_clip_limit, clahe_grid_size, noise_reduction, min_marker_perimeter_rate, poly_approx_accuracy_rate, max_erroneous_bits_border, error_correction_rate, current_camera_index, requested_camera_index, camera_changed, is_switching_camera, cap
    global roi_tracking_enabled, bilateral_filtering, bilateral_d, bilateral_sigma_color, bilateral_sigma_space, morphology_enabled, morphology_kernel_size, kalman_enabled, kalman_q, kalman_r, roi_active, roi_box, kalman_predict_count, last_measurement_time, tracking_mode
    
    # Try opening webcam indices 0, 1, 2 on startup (using DSHOW first)
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
        print("Camera Tracker WARNING: No webcam could be opened! Emulating offline status.")
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Frame dimensions
    width = 640
    height = 480

    print("Camera Tracker: Tracker loop started.")
    
    while True:
        loop_start = time.time()
        
        # Check if camera change was requested
        if camera_changed:
            camera_changed = False
            if not is_switching_camera:
                is_switching_camera = True
                # Run the open camera logic in a separate background thread!
                # This guarantees that if cv2.VideoCapture hangs, the main loop remains alive.
                thread = threading.Thread(target=async_open_camera, args=(requested_camera_index,), daemon=True)
                thread.start()
        
        # If camera is switching or not ready yet, display connecting screen
        if is_switching_camera or cap is None:
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(dummy_frame, f"Cargando Camara {requested_camera_index}...", (160, 220), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 180, 50), 2, cv2.LINE_AA)
            cv2.putText(dummy_frame, "Por favor espera, iniciando dispositivo de video...", (110, 260), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
            
            _, buffer = cv2.imencode('.jpg', dummy_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            with data_lock:
                latest_data = {
                    "detected": False,
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "pitch": 0.0,
                    "yaw": 0.0,
                    "roll": 0.0,
                    "rotation_matrix": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    "frame": frame_base64
                }
            time.sleep(0.1)
            continue
            
        # Capture frame
        ret = False
        frame = None
        if cap is not None:
            try:
                ret, frame = cap.read()
            except Exception as e:
                print(f"Error reading from camera: {e}")
                ret = False
            
        if ret and frame is not None:
            # Get actual frame size in case it is different from requested
            height, width = frame.shape[:2]
            
            # Build approximate camera intrinsic matrix
            f_px = width * focal_length_factor
            K = np.array([
                [f_px, 0.0, width / 2.0],
                [0.0, f_px, height / 2.0],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)
            
            # Assume no lens distortion
            dist_coeffs = np.zeros((4, 1), dtype=np.float32)
            
            # Preprocess the frame for ArUco detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply bilateral filter (if enabled) to preserve edges and reduce noise/fog
            if bilateral_filtering:
                gray = cv2.bilateralFilter(gray, bilateral_d, bilateral_sigma_color, bilateral_sigma_space)
            
            # Apply brightness and contrast (manual)
            gray = cv2.convertScaleAbs(gray, alpha=contrast, beta=brightness)
            
            # Apply CLAHE (auto-contrast) if enabled
            if auto_contrast:
                grid = max(2, clahe_grid_size)
                clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(grid, grid))
                gray = clahe.apply(gray)
            
            # Apply Median Blur for noise reduction
            if noise_reduction > 0:
                ksize = int(noise_reduction)
                if ksize % 2 == 0:
                    ksize += 1
                gray = cv2.medianBlur(gray, ksize)
            
            # Apply Morphological closing/opening to heal spots (dirt/reflections)
            if morphology_enabled:
                k_size = int(morphology_kernel_size)
                if k_size % 2 == 0:
                    k_size += 1
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
                gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
                gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            
            # Update detector parameters dynamically in real-time!
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
                
            # Determine display_frame (what we draw on and send to client)
            if show_processed:
                # Get the thresholded binary frame to show exactly what ArUco sees!
                if hasattr(cv2.aruco, 'getPredefinedDictionary'):
                    block_size = int(detector.getDetectorParameters().adaptiveThreshWinSizeMax)
                else:
                    block_size = int(parameters.adaptiveThreshWinSizeMax)
                if block_size % 2 == 0:
                    block_size += 1
                block_size = max(3, block_size)
                
                thresh = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                    cv2.THRESH_BINARY, block_size, 7
                )
                display_frame = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            else:
                # Send the color frame with manual brightness/contrast applied
                display_frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
            
            # Detect ArUco markers (either ROI or Full Frame)
            corners = []
            ids = None
            rejected = []
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
            pitch = yaw = roll = 0.0
            R_flat = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            tracked_marker = -1
            kalman_active = False
            
            if ids is not None and len(ids) > 0:
                ids_flat = ids.flatten()
                
                # Check for target markers based on tracking_mode
                idx = -1
                size_phys = marker_size
                offset_local = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)
                
                if tracking_mode == "single_small":
                    for i, mid in enumerate(ids_flat):
                        if mid == 1:
                            idx = i
                            tracked_marker = 1
                            size_phys = marker_size  # Standalone testing, use full marker_size
                            offset_local = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)  # Centered
                            break
                elif tracking_mode == "single_large":
                    for i, mid in enumerate(ids_flat):
                        if mid == 0:
                            idx = i
                            tracked_marker = 0
                            size_phys = marker_size  # Standalone testing, use full marker_size
                            offset_local = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)  # Centered
                            break
                else:  # "dual" (default)
                    # Prioritize ID 1 (Small)
                    for i, mid in enumerate(ids_flat):
                        if mid == 1:
                            idx = i
                            tracked_marker = 1
                            size_phys = marker_size / 5.0
                            offset_local = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)  # Centered landing pad
                            break
                    
                    if idx == -1:
                        for i, mid in enumerate(ids_flat):
                            if mid == 0:
                                idx = i
                                tracked_marker = 0
                                size_phys = marker_size
                                # Y-offset for large marker (Y=170 vs Y=400 in 800px grid is +230px, physical offset is 230/300 of large marker size)
                                offset_local = np.array([[0.0], [marker_size * (230.0 / 300.0)], [0.0]], dtype=np.float32)
                                break
                
                if idx != -1:
                    img_points = corners[idx][0].astype(np.float32)
                        
                    # 3D coordinates of the marker corners in its own frame (in meters)
                    obj_points = np.array([
                        [-size_phys/2.0, size_phys/2.0, 0.0],
                        [size_phys/2.0, size_phys/2.0, 0.0],
                        [size_phys/2.0, -size_phys/2.0, 0.0],
                        [-size_phys/2.0, -size_phys/2.0, 0.0]
                    ], dtype=np.float32)
                    
                    # Solve PnP for 3D Pose
                    try:
                        success, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                    except Exception:
                        success, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, dist_coeffs)
                        
                    if success:
                        detected = True
                        R, _ = cv2.Rodrigues(rvec)
                        
                        # Translate marker coordinates to target landing pad center coordinates
                        tvec_landing = tvec + R @ offset_local
                        
                        tx = float(tvec_landing[0][0])
                        ty = float(tvec_landing[1][0])
                        tz = float(tvec_landing[2][0])
                        rx = float(rvec[0][0])
                        ry = float(rvec[1][0])
                        rz = float(rvec[2][0])
                        
                        if kalman_enabled:
                            now = time.time()
                            if last_measurement_time is None:
                                dt = 1.0 / 30.0
                            else:
                                dt = now - last_measurement_time
                            last_measurement_time = now
                            
                            if kalman_predict_count > 0:
                                reset_kalman(tx, ty, tz, rx, ry, rz)
                                kalman_predict_count = 0
                            else:
                                kf_x.predict(dt)
                                kf_y.predict(dt)
                                kf_z.predict(dt)
                                kf_rx.predict(dt)
                                kf_ry.predict(dt)
                                kf_rz.predict(dt)
                                
                                kf_x.update(tx)
                                kf_y.update(ty)
                                kf_z.update(tz)
                                kf_rx.update(rx)
                                kf_ry.update(ry)
                                kf_rz.update(rz)
                                
                            tx = kf_x.x
                            ty = kf_y.x
                            tz = kf_z.x
                            rx = kf_rx.x
                            ry = kf_ry.x
                            rz = kf_rz.x
                            
                            rvec_smooth = np.array([[rx], [ry], [rz]], dtype=np.float32)
                            R, _ = cv2.Rodrigues(rvec_smooth)
                            tvec_drawn = tvec_landing - R @ offset_local
                        else:
                            tvec_drawn = tvec
                            rvec_smooth = rvec
                            
                        R_flat = R.flatten().tolist()
                        pitch, yaw, roll = get_euler_angles(R)
                        
                        # ROI bounding box calculation for next frame
                        if roi_tracking_enabled:
                            c_pts = corners[idx][0]
                            x_min = np.min(c_pts[:, 0])
                            x_max = np.max(c_pts[:, 0])
                            y_min = np.min(c_pts[:, 1])
                            y_max = np.max(c_pts[:, 1])
                            w = x_max - x_min
                            h = y_max - y_min
                            cx = (x_min + x_max) / 2
                            cy = (y_min + y_max) / 2
                            
                            roi_x1 = max(0, int(cx - w * 1.5))
                            roi_y1 = max(0, int(cy - h * 1.5))
                            roi_x2 = min(width, int(cx + w * 1.5))
                            roi_y2 = min(height, int(cy + h * 1.5))
                            roi_box = [roi_x1, roi_y1, roi_x2, roi_y2]
                            roi_active = True
                        
                        # Draw ArUco border on display frame
                        cv2.aruco.drawDetectedMarkers(display_frame, [corners[idx]], np.array([[tracked_marker]]))
                        draw_axes(display_frame, K, dist_coeffs, rvec_smooth, tvec_drawn, size_phys * 0.5)
                        
                        # Draw ROI boundary box (green)
                        if roi_tracking_enabled and roi_active:
                            cv2.rectangle(display_frame, (roi_box[0], roi_box[1]), (roi_box[2], roi_box[3]), (0, 255, 0), 2)
            
            if not detected:
                # Kalman prediction during dropout
                if kalman_enabled and last_measurement_time is not None and kalman_predict_count < MAX_KALMAN_PREDICT_FRAMES:
                    now = time.time()
                    dt = now - last_measurement_time
                    last_measurement_time = now
                    kalman_predict_count += 1
                    
                    kf_x.predict(dt)
                    kf_y.predict(dt)
                    kf_z.predict(dt)
                    kf_rx.predict(dt)
                    kf_ry.predict(dt)
                    kf_rz.predict(dt)
                    
                    tx = kf_x.x
                    ty = kf_y.x
                    tz = kf_z.x
                    rx = kf_rx.x
                    ry = kf_ry.x
                    rz = kf_rz.x
                    
                    rvec_smooth = np.array([[rx], [ry], [rz]], dtype=np.float32)
                    R, _ = cv2.Rodrigues(rvec_smooth)
                    R_flat = R.flatten().tolist()
                    pitch, yaw, roll = get_euler_angles(R)
                    
                    detected = True
                    kalman_active = True
                    
                    cv2.putText(display_frame, "KALMAN ESTIMATING...", (15, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)
                else:
                    roi_active = False
                    if kalman_predict_count >= MAX_KALMAN_PREDICT_FRAMES:
                        last_measurement_time = None
            
            # Compress and encode display frame as base64 JPEG
            _, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Update global state
            with data_lock:
                latest_data = {
                    "detected": detected,
                    "x": tx,
                    "y": ty,
                    "z": tz,
                    "pitch": pitch,
                    "yaw": yaw,
                    "roll": roll,
                    "rotation_matrix": R_flat,
                    "tracked_marker": tracked_marker,
                    "kalman_active": kalman_active,
                    "frame": frame_base64
                }
                
        else:
            # If camera capture fails or is offline
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(dummy_frame, "Webcam Offline / In Use", (140, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 220), 2, cv2.LINE_AA)
            cv2.putText(dummy_frame, "Make sure your webcam is plugged in and not used by another app.", (20, 280), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
            
            _, buffer = cv2.imencode('.jpg', dummy_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            with data_lock:
                latest_data = {
                    "detected": False,
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "pitch": 0.0,
                    "yaw": 0.0,
                    "roll": 0.0,
                    "rotation_matrix": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    "frame": frame_base64
                }
        
        # Enforce ~30 FPS loop rate
        elapsed = time.time() - loop_start
        sleep_time = max(1.0/30.0 - elapsed, 0)
        time.sleep(sleep_time)

if __name__ == "__main__":
    # Scan cameras first
    scan_available_cameras()
    
    # Start HTTP Server thread
    http_thread = threading.Thread(target=run_http_server, daemon=True)
    http_thread.start()
    
    # Start WebSocket Server thread
    ws_thread = threading.Thread(target=run_ws_server, daemon=True)
    ws_thread.start()
    
    # Start Camera Tracking loop in the main thread
    try:
        run_camera_tracking()
    except KeyboardInterrupt:
        print("\nStopping servers...")
