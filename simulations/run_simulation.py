"""
Script Maestro de Simulación SkyDock.
Ejecuta la simulación completa del cuadricóptero de 8 pulgadas, cámara nadir OV9281,
objetivo visual (ArUco / QR) y la computadora a bordo Radxa Zero 3W con MAVLink.
"""
import sys
import os
import time
import argparse
from pathlib import Path

# Añadir la raíz del repositorio a sys.path para permitir ejecución directa o como paquete
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from simulations.config.drone_params import DRONE_PARAMS
from simulations.config.camera_params import CAMERA_PARAMS
from simulations.vision.sim_camera import SyntheticNadirCamera
from simulations.sitl.quad_sitl import Quad8InchSITL
from simulations.companion.companion_node import RadxaCompanionNode
from simulations.companion.landing_controller import LandingState
from simulations.dashboard.sim_dashboard import SimulationDashboard

def parse_args():
    parser = argparse.ArgumentParser(description="Simulación SkyDock 8-Inch Quad + Radxa Zero 3W + OV9281")
    parser.add_argument("--altitude", type=float, default=3.0, help="Altitud inicial de prueba (metros)")
    parser.add_argument("--offset_x", type=float, default=1.2, help="Desplazamiento inicial X (Norte) respecto a la plataforma")
    parser.add_argument("--offset_y", type=float, default=-0.8, help="Desplazamiento inicial Y (Este) respecto a la plataforma")
    parser.add_argument("--yaw", type=float, default=20.0, help="Ángulo inicial de guiñada del dron (grados)")
    parser.add_argument("--target_type", type=str, default="aruco", choices=["aruco", "qr"], help="Tipo de objetivo visual")
    parser.add_argument("--marker_id", type=int, default=0, help="ID del marcador ArUco")
    parser.add_argument("--wind_x", type=float, default=0.0, help="Viento constante en X (m/s)")
    parser.add_argument("--wind_y", type=float, default=0.0, help="Viento constante en Y (m/s)")
    parser.add_argument("--headless", action="store_true", help="Ejecutar en modo consola sin interfaz gráfica")
    parser.add_argument("--auto_start", action="store_true", default=False, help="Iniciar aterrizaje autónomo automáticamente al arrancar")
    return parser.parse_args()

def main():
    args = parse_args()
    print("=" * 70)
    print(" INICIANDO SIMULADOR SKYDOCK: CUADRICÓPTERO 8\" + RADXA ZERO 3W + OV9281")
    print("=" * 70)

    # 1. Inicializar Simulador de Cámara Nadir OV9281
    camera = SyntheticNadirCamera(
        params=CAMERA_PARAMS,
        target_type=args.target_type,
        marker_id=args.marker_id,
        marker_size_m=0.055,
        pad_size_m=0.80
    )
    camera.set_pad_position(0.0, 0.0, 0.0)

    # 2. Inicializar Simulador de Física 6-DOF y MAVLink SITL
    sitl = Quad8InchSITL(params=DRONE_PARAMS, bind_host="127.0.0.1", bind_port=14550)
    # Posicionar dron en el aire con el offset especificado (z negativo para altitud)
    initial_pos = np.array([args.offset_x, args.offset_y, -args.altitude], dtype=np.float64)
    sitl.reset(initial_pos_ned=initial_pos, initial_yaw_deg=args.yaw)
    sitl.wind_ned = np.array([args.wind_x, args.wind_y, 0.0], dtype=np.float64)
    sitl.start()
    
    # Armar y poner en modo GUIDED por defecto para la simulación
    sitl.armed = True
    sitl.flight_mode = "GUIDED"

    # 3. Inicializar Nodo de la Companion Computer (Radxa Zero 3W)
    radxa = RadxaCompanionNode(
        mavlink_conn_str="udpout:127.0.0.1:14550",
        enable_csv_logging=False,
        target_loop_rate_hz=60.0
    )

    # Conectar el proveedor de imágenes de la cámara sintética a la Radxa
    def camera_frame_provider():
        state = sitl.get_state()
        pos_ned = state["pos_ned"]
        euler_rad = state["euler_rad"]
        return camera.render_frame(pos_ned, euler_rad, add_sensor_noise=True)

    radxa.set_frame_provider(camera_frame_provider)
    radxa.start(auto_connect_mavlink=True)

    # Si auto_start está activo, activar secuencia de aterrizaje tras 1 segundo
    if args.auto_start:
        time.sleep(0.5)
        radxa.trigger_landing_sequence()

    # 4. Inicializar Dashboard
    dashboard = SimulationDashboard() if not args.headless else None
    if dashboard is not None:
        cv2.namedWindow("SkyDock Autonomous Simulation", cv2.WINDOW_NORMAL)

    print("\n[SISTEMA LISTO] Presiona [ESC] en la ventana o Ctrl+C en la terminal para detener.")
    print("Controles: [G]=Grabar Ensayo, [SPACE]=Aterrizar, [T]=Takeoff 3m, [R]=Reset, [W/A/S/D]=Mover Dron\n")

    t_prev = time.time()
    fps_display = 60.0

    try:
        while True:
            t_loop = time.time()
            dt = t_loop - t_prev
            t_prev = t_loop
            fps_display = (0.9 * fps_display) + (0.1 * (1.0 / max(dt, 0.001)))

            # Obtener estados actuales
            sitl_state = sitl.get_state()
            vision_data = radxa.last_detection_data
            radxa_state = {
                "state": radxa.last_state.value,
                "cmd_vel": radxa.last_cmd_vel,
                "is_recording": radxa.is_recording_trial,
                "trial_num": radxa.current_trial_num
            }

            # Generar frame actual de la cámara para visualización
            current_frame = camera_frame_provider()

            # Renderizar Dashboard si no está en modo headless
            if dashboard is not None:
                dash_view = dashboard.render(
                    camera_frame=current_frame,
                    vision_data=vision_data,
                    sitl_state=sitl_state,
                    radxa_state=radxa_state,
                    camera_params=CAMERA_PARAMS,
                    fps_sim=fps_display
                )

                cv2.imshow("SkyDock Autonomous Simulation", dash_view)
                key = cv2.waitKey(15) & 0xFF

                if key == 27: # ESC para Salir
                    break
                elif key in [ord('g'), ord('G')]: # Toggle Grabación de Ensayo
                    is_rec = radxa.toggle_trial_recording()
                elif key == 32: # SPACE (Iniciar aterrizaje autónomo)
                    radxa.trigger_landing_sequence()
                elif key in [ord('q'), ord('Q')]: # Girar Yaw Izquierda (CCW)
                    sitl.euler[2] -= np.radians(8.0)
                    sitl.euler[2] = (sitl.euler[2] + np.pi) % (2 * np.pi) - np.pi
                elif key in [ord('e'), ord('E')]: # Girar Yaw Derecha (CW)
                    sitl.euler[2] += np.radians(8.0)
                    sitl.euler[2] = (sitl.euler[2] + np.pi) % (2 * np.pi) - np.pi
                elif key in [ord('v'), ord('V')]: # Toggle View Mode (Dual / 3D / Nadir)
                    dashboard.toggle_view_mode()
                elif key in [ord('c'), ord('C')]: # Toggle 3D Cam Mode (Chase / Orbit)
                    dashboard.toggle_3d_cam_mode()
                elif key in [ord('t'), ord('T')]: # Takeoff y mantener en hover (IDLE)
                    sitl.flight_mode = "GUIDED"
                    sitl.pos[2] = -3.0
                    sitl.vel = np.zeros(3)
                    sitl.target_vel_body = np.zeros(3)
                    radxa.controller.set_state(LandingState.IDLE)
                    radxa.last_cmd_vel = np.zeros(3)
                elif key in [ord('r'), ord('R')]: # Reset a posición inicial en hover (IDLE)
                    sitl.reset(initial_pos_ned=initial_pos, initial_yaw_deg=args.yaw)
                    sitl.armed = True
                    sitl.flight_mode = "GUIDED"
                    sitl.target_vel_body = np.zeros(3)
                    radxa.controller.set_state(LandingState.IDLE)
                    radxa.last_cmd_vel = np.zeros(3)
                elif key in [ord('l'), ord('L')]: # LAND
                    sitl.flight_mode = "LAND"
                    radxa.controller.set_state(LandingState.TOUCHDOWN)
                elif key in [ord('w'), ord('W')]: # Empujar Norte
                    sitl.pos[0] += 0.3
                elif key in [ord('s'), ord('S')]: # Empujar Sur
                    sitl.pos[0] -= 0.3
                elif key in [ord('a'), ord('A')]: # Empujar Oeste
                    sitl.pos[1] -= 0.3
                elif key in [ord('d'), ord('D')]: # Empujar Este
                    sitl.pos[1] += 0.3
            else:
                # Modo Headless: Imprimir telemetría en terminal
                pos = sitl_state["pos_ned"]
                alt = sitl_state["altitude_m"]
                state_str = radxa_state["state"]
                det = vision_data.get("detected", False)
                print(f"[SIM] Alt: {alt:.2f}m | Pos: ({pos[0]:.2f}, {pos[1]:.2f}) | State: {state_str:10s} | Detected: {det} | FPS: {fps_display:.1f}", end='\r')
                time.sleep(0.03)

    except KeyboardInterrupt:
        print("\nInterrupción recibida...")

    finally:
        print("\nCerrando simulación...")
        radxa.stop()
        sitl.stop()
        if dashboard is not None:
            cv2.destroyAllWindows()
        print("Simulación finalizada con éxito.")

if __name__ == "__main__":
    main()
