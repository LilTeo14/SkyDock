"""
Panel de Control y Visualizador en Tiempo Real del Sistema SkyDock.
Diseño optimizado:
1. Parte Superior (Pantalla Principal): Cámara Nadir OV9281 (POV) y Vista 3D en Tercera Persona lado a lado a máxima resolución.
2. Parte Inferior Izquierda/Centro: Cuadro compacto de telemetría, cinemática, visión y comandos MAVLink.
3. Parte Inferior Derecha: Radar 2D compacto de aproximación y aterrizaje.
"""
import cv2
import numpy as np
import time
from typing import Dict, Any, Optional

from simulations.vision.third_person_viewer import ThirdPersonViewer3D

class SimulationDashboard:
    def __init__(self, window_name: str = "SkyDock 8-Inch Quad Simulation - Radxa Zero 3W + OV9281"):
        self.window_name = window_name
        self.dash_width = 1580
        self.dash_height = 840
        
        # Renderizador 3D en tercera persona
        self.viewer_3d = ThirdPersonViewer3D(width=760, height=480, fov_deg=65.0)
        
        # Modos de visualización: "DUAL" (ambas cámaras), "POV_ONLY" (solo nadir), "CHASE_ONLY" (solo 3D)
        self.view_mode = "DUAL"
        
        # Historial de radar compacto
        self.trajectory_radar = []
        self.max_trail = 90

    def toggle_view_mode(self):
        """Alterna entre DUAL, CHASE_ONLY y POV_ONLY."""
        modes = ["DUAL", "CHASE_ONLY", "POV_ONLY"]
        idx = (modes.index(self.view_mode) + 1) % len(modes)
        self.view_mode = modes[idx]
        print(f"[DASHBOARD] Modo de visualización: {self.view_mode}")

    def toggle_3d_cam_mode(self):
        """Alterna la cámara 3D entre Chase Cam y Orbit Cam."""
        self.viewer_3d.toggle_camera_mode()

    def render(self, camera_frame: np.ndarray,
               vision_data: Dict[str, Any],
               sitl_state: Dict[str, Any],
               radxa_state: Dict[str, Any],
               camera_params,
               fps_sim: float = 0.0) -> np.ndarray:
        """
        Compone la vista global del dashboard.
        """
        canvas = np.ones((self.dash_height, self.dash_width, 3), dtype=np.uint8) * 22 # Fondo oscuro

        pos_ned = sitl_state.get("pos_ned", np.zeros(3))
        euler_rad = sitl_state.get("euler_rad", np.zeros(3))
        detected = vision_data.get("detected", False)
        corners = vision_data.get("corners")
        rvec = vision_data.get("rvec")
        tvec = vision_data.get("tvec_cam")

        # -------------------------------------------------------------
        # 1. PREPARAR FEED DE CÁMARA NADIR OV9281 CON OVERLAYS
        # -------------------------------------------------------------
        if len(camera_frame.shape) == 2:
            cam_bgr = cv2.cvtColor(camera_frame, cv2.COLOR_GRAY2BGR)
        else:
            cam_bgr = camera_frame.copy()

        h_cam, w_cam = cam_bgr.shape[:2]
        cx, cy = w_cam // 2, h_cam // 2

        # Retícula central óptica
        cv2.line(cam_bgr, (cx - 20, cy), (cx + 20, cy), (0, 255, 255), 1)
        cv2.line(cam_bgr, (cx, cy - 20), (cx, cy + 20), (0, 255, 255), 1)
        cv2.circle(cam_bgr, (cx, cy), 14, (0, 255, 255), 1)

        # Overlays de marcador
        if detected and corners is not None:
            pts = corners.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(cam_bgr, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            m_center = np.mean(corners.reshape(-1, 2), axis=0).astype(int)
            cv2.circle(cam_bgr, tuple(m_center), 4, (0, 0, 255), -1)
            cv2.line(cam_bgr, (cx, cy), tuple(m_center), (0, 165, 255), 2)

            if rvec is not None and tvec is not None:
                try:
                    cv2.drawFrameAxes(cam_bgr, camera_params.camera_matrix,
                                      camera_params.distortion_coeffs,
                                      rvec, tvec, 0.08, 2)
                except Exception:
                    pass

        # -------------------------------------------------------------
        # 2. RENDERIZAR VISTA 3D EN TERCERA PERSONA
        # -------------------------------------------------------------
        view_3d_img = self.viewer_3d.render(
            drone_pos_ned=pos_ned,
            drone_euler_rad=euler_rad,
            pad_world_pos=np.array([0.0, 0.0, 0.0]),
            target_detected=detected
        )

        # -------------------------------------------------------------
        # 3. SECCIÓN SUPERIOR: PANTALLAS DE VÍDEO
        # -------------------------------------------------------------
        top_y = 12
        top_h = 495
        left_margin = 16
        gap = 12

        is_rec = radxa_state.get("is_recording", False)
        trial_num = radxa_state.get("trial_num", 0)

        if self.view_mode == "DUAL":
            # Dos cámaras lado a lado calculadas exactamente
            pane_w = (self.dash_width - (left_margin * 2) - gap) // 2
            
            # Cámara Nadir (Izquierda)
            cam_resized = cv2.resize(cam_bgr, (pane_w, top_h))
            canvas[top_y:top_y + top_h, left_margin:left_margin + pane_w] = cam_resized
            cv2.rectangle(canvas, (left_margin, top_y), (left_margin + pane_w, top_y + top_h), (75, 75, 75), 2)
            
            status_col = (0, 255, 0) if detected else (0, 140, 255)
            status_txt = "OV9281 NADIR (TARGET LOCK)" if detected else "OV9281 NADIR (SEARCHING)"
            cv2.putText(canvas, status_txt, (left_margin + 15, top_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_col, 2)

            # Badge de Grabación en Nadir (Top-Right)
            if is_rec:
                rec_badge = f"[● REC: ENSAYO #{trial_num:03d}]"
                rec_color = (0, 0, 255) # Rojo vivo
            else:
                rec_badge = "[○ MODO LIBRE (NO GRABANDO)]"
                rec_color = (180, 180, 180) # Gris neutro
            cv2.putText(canvas, rec_badge, (left_margin + pane_w - 280, top_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.46, rec_color, 2)

            # Vista 3D (Derecha)
            right_x = left_margin + pane_w + gap
            v3d_resized = cv2.resize(view_3d_img, (pane_w, top_h))
            canvas[top_y:top_y + top_h, right_x:right_x + pane_w] = v3d_resized
            cv2.rectangle(canvas, (right_x, top_y), (right_x + pane_w, top_y + top_h), (75, 75, 75), 2)

        elif self.view_mode == "CHASE_ONLY":
            full_w = self.dash_width - (left_margin * 2)
            v3d_resized = cv2.resize(view_3d_img, (full_w, top_h))
            canvas[top_y:top_y + top_h, left_margin:left_margin + full_w] = v3d_resized
            cv2.rectangle(canvas, (left_margin, top_y), (left_margin + full_w, top_y + top_h), (75, 75, 75), 2)

        elif self.view_mode == "POV_ONLY":
            full_w = self.dash_width - (left_margin * 2)
            cam_resized = cv2.resize(cam_bgr, (full_w, top_h))
            canvas[top_y:top_y + top_h, left_margin:left_margin + full_w] = cam_resized
            cv2.rectangle(canvas, (left_margin, top_y), (left_margin + full_w, top_y + top_h), (75, 75, 75), 2)

        # -------------------------------------------------------------
        # 4. SECCIÓN INFERIOR: TELEMETRÍA COMPACTA + RADAR 2D (DERECHA)
        # -------------------------------------------------------------
        bot_y = 520
        bot_h = 285
        radar_w = 240
        telem_w = self.dash_width - (left_margin * 2) - radar_w - gap

        # Caja de Telemetría Compacta
        cv2.rectangle(canvas, (left_margin, bot_y), (left_margin + telem_w, bot_y + bot_h), (28, 28, 28), -1)
        cv2.rectangle(canvas, (left_margin, bot_y), (left_margin + telem_w, bot_y + bot_h), (60, 60, 60), 1)

        # 4 Columnas de Telemetría Compacta
        col_w = telem_w // 4
        
        def draw_row(col_idx, row_y, label, value, val_color=(255, 255, 255)):
            x_start = left_margin + (col_idx * col_w) + 16
            cv2.putText(canvas, label, (x_start, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)
            cv2.putText(canvas, str(value), (x_start + 145, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, val_color, 1)

        def draw_header(col_idx, header_text):
            x_start = left_margin + (col_idx * col_w) + 16
            cv2.putText(canvas, header_text, (x_start, bot_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 2)
            cv2.line(canvas, (x_start, bot_y + 36), (x_start + col_w - 30, bot_y + 36), (65, 65, 65), 1)

        # --- Columna 0: Estado del Vehículo ---
        draw_header(0, "ESTADO GENERAL")
        fmode = sitl_state.get("flight_mode", "N/A")
        armed = sitl_state.get("armed", False)
        lstate = radxa_state.get("state", "IDLE")
        
        y_r = bot_y + 65
        draw_row(0, y_r, "Flight Mode:", fmode, (0, 255, 255))
        y_r += 24
        draw_row(0, y_r, "Motors:", "ARMED" if armed else "DISARMED", (0, 255, 0) if armed else (0, 0, 255))
        y_r += 24
        draw_row(0, y_r, "Radxa State:", lstate, (255, 180, 0))
        y_r += 24
        draw_row(0, y_r, "Loop Rate:", f"{fps_sim:.1f} FPS")
        y_r += 24
        draw_row(0, y_r, "Landing Legs:", "10 cm CLEARANCE", (180, 220, 180))

        # --- Columna 1: Cinemática y Altitud ---
        draw_header(1, "CINEMÁTICA & AGL")
        alt = sitl_state.get("altitude_m", 0.0)
        vel = sitl_state.get("vel_ned", np.zeros(3))
        
        y_r = bot_y + 65
        draw_row(1, y_r, "Altitude (AGL):", f"{alt:.2f} m", (255, 255, 255))
        y_r += 24
        draw_row(1, y_r, "Horiz Speed:", f"{np.hypot(vel[0], vel[1]):.2f} m/s")
        y_r += 24
        draw_row(1, y_r, "Climb/Desc:", f"{-vel[2]:.2f} m/s", (0, 255, 0) if vel[2] > 0 else (255, 255, 255))
        y_r += 24
        draw_row(1, y_r, "Roll / Pitch:", f"{np.degrees(euler_rad[0]):+.1f}° / {np.degrees(euler_rad[1]):+.1f}°")
        y_r += 24
        draw_row(1, y_r, "Yaw Heading:", f"{np.degrees(euler_rad[2]):.1f}°")

        # --- Columna 2: Percepción Radxa (OV9281) ---
        draw_header(2, "PERCEPCIÓN RADXA")
        pos_b = vision_data.get("pos_body_filtered")
        y_r = bot_y + 65
        if pos_b is not None:
            draw_row(2, y_r, "Target X (Fwd):", f"{pos_b[0]:+.3f} m")
            y_r += 24
            draw_row(2, y_r, "Target Y (Rgt):", f"{pos_b[1]:+.3f} m")
            y_r += 24
            draw_row(2, y_r, "Target Z (Dist):", f"{pos_b[2]:.3f} m")
            y_r += 24
            horiz_err = np.hypot(pos_b[0], pos_b[1])
            draw_row(2, y_r, "Radial Error:", f"{horiz_err:.3f} m", (0, 255, 0) if horiz_err < 0.12 else (0, 165, 255))
            y_r += 24
            draw_row(2, y_r, "Kalman Filter:", "LOCKED (3D)", (0, 255, 0))
        else:
            draw_row(2, y_r, "Target Status:", "SEARCHING LOCK", (0, 140, 255))
            y_r += 24
            draw_row(2, y_r, "Target X (Fwd):", "--")
            y_r += 24
            draw_row(2, y_r, "Target Y (Rgt):", "--")
            y_r += 24
            draw_row(2, y_r, "Target Z (Dist):", "--")
            y_r += 24
            draw_row(2, y_r, "Kalman Filter:", "COASTING", (160, 160, 160))

        # --- Columna 3: Comandos MAVLink Radxa ---
        draw_header(3, "COMANDOS MAVLINK")
        cmd_vel = radxa_state.get("cmd_vel", np.zeros(3))
        y_r = bot_y + 65
        draw_row(3, y_r, "Cmd Vx (Fwd):", f"{cmd_vel[0]:+.2f} m/s")
        y_r += 24
        draw_row(3, y_r, "Cmd Vy (Rgt):", f"{cmd_vel[1]:+.2f} m/s")
        y_r += 24
        draw_row(3, y_r, "Cmd Vz (Down):", f"{cmd_vel[2]:+.2f} m/s")
        y_r += 24
        draw_row(3, y_r, "Display Mode:", self.view_mode, (0, 255, 255))
        y_r += 24
        draw_row(3, y_r, "3D Cam Type:", self.viewer_3d.cam_mode, (0, 255, 255))

        # -------------------------------------------------------------
        # 5. RADAR 2D COMPACTO (Abajo a la Derecha)
        # -------------------------------------------------------------
        radar_x = left_margin + telem_w + 14
        cv2.rectangle(canvas, (radar_x, bot_y), (radar_x + radar_w, bot_y + bot_h), (28, 28, 28), -1)
        cv2.rectangle(canvas, (radar_x, bot_y), (radar_x + radar_w, bot_y + bot_h), (60, 60, 60), 1)

        cv2.putText(canvas, "RADAR 2D", (radar_x + 85, bot_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 2)

        radar_cx = radar_x + (radar_w // 2)
        radar_cy = bot_y + 135
        radar_r = 95

        cv2.circle(canvas, (radar_cx, radar_cy), radar_r, (38, 38, 38), -1)
        cv2.circle(canvas, (radar_cx, radar_cy), radar_r, (75, 75, 75), 1)

        # Anillos 2m, 1m, 0.5m
        scale_px_m = 42
        for r_m, lbl in [(2.0, "2m"), (1.0, "1m"), (0.5, "0.5m")]:
            r_px = int(r_m * scale_px_m)
            if r_px < radar_r:
                cv2.circle(canvas, (radar_cx, radar_cy), r_px, (55, 55, 55), 1)
                cv2.putText(canvas, lbl, (radar_cx + r_px - 18, radar_cy - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (110, 110, 110), 1)

        cv2.line(canvas, (radar_cx - radar_r, radar_cy), (radar_cx + radar_r, radar_cy), (55, 55, 55), 1)
        cv2.line(canvas, (radar_cx, radar_cy - radar_r), (radar_cx, radar_cy + radar_r), (55, 55, 55), 1)

        # Plataforma SkyDock en centro del radar
        cv2.rectangle(canvas, (radar_cx - 7, radar_cy - 7), (radar_cx + 7, radar_cy + 7), (0, 200, 200), -1)

        # Posición del dron en el radar
        rx_px = int(radar_cx + pos_ned[1] * scale_px_m)
        ry_px = int(radar_cy - pos_ned[0] * scale_px_m)
        
        # Limitar dentro del radio del radar
        dist_from_center = np.hypot(rx_px - radar_cx, ry_px - radar_cy)
        if dist_from_center > radar_r - 4:
            angle = np.arctan2(ry_px - radar_cy, rx_px - radar_cx)
            rx_px = int(radar_cx + (radar_r - 4) * np.cos(angle))
            ry_px = int(radar_cy + (radar_r - 4) * np.sin(angle))

        self.trajectory_radar.append((rx_px, ry_px))
        if len(self.trajectory_radar) > self.max_trail:
            self.trajectory_radar.pop(0)

        for i in range(1, len(self.trajectory_radar)):
            cv2.line(canvas, self.trajectory_radar[i-1], self.trajectory_radar[i], (0, 130, 255), 1)

        cv2.circle(canvas, (rx_px, ry_px), 5, (0, 0, 255), -1)
        cv2.putText(canvas, "QUAD", (rx_px + 7, ry_px + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1)

        # Coordenadas numéricas relativas en el radar
        cv2.putText(canvas, f"X:{pos_ned[0]:+.2f}m Y:{pos_ned[1]:+.2f}m", (radar_x + 45, bot_y + bot_h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

        # -------------------------------------------------------------
        # 6. BARRA INFERIOR DE ATAJOS Y CONTROLES
        # -------------------------------------------------------------
        cv2.putText(canvas, "[G]: Grabar Ensayo   [SPACE]: Aterrizar   [Q/E]: Girar Yaw   [W/A/S/D]: Mover   [V]: Vistas   [C]: Chase/Orbit   [T]: Takeoff   [R]: Reset   [ESC]: Salir",
                    (left_margin + 10, self.dash_height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (190, 190, 190), 1)

        return canvas
