"""
Cargador y procesador del modelo 3D CAD BUMBLEBEE.obj para el visor en tercera persona.
Transforma las coordenadas del CAD (Autodesk ATF mm) al marco del cuerpo del dron (FRD en metros)
y genera una representación optimizada en tiempo real a 60 FPS con colores distintivos.
"""
import os
import time
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import cv2

class BumblebeeMeshModel:
    def __init__(self, obj_path: Optional[str] = None, max_edges: int = 3200):
        self.max_edges = max_edges
        self.is_loaded = False
        
        if obj_path is None:
            # Buscar BUMBLEBEE.obj en raíz del proyecto o en simulations/
            proj_root = Path(__file__).resolve().parent.parent.parent
            candidates = [
                proj_root / "BUMBLEBEE.obj",
                proj_root / "simulations" / "BUMBLEBEE.obj",
                proj_root / "simulations" / "models" / "BUMBLEBEE.obj"
            ]
            for cand in candidates:
                if cand.exists():
                    obj_path = str(cand)
                    break
        
        self.obj_path = obj_path
        self.v1_body = np.zeros((0, 3), dtype=np.float32)
        self.v2_body = np.zeros((0, 3), dtype=np.float32)
        self.edge_colors = []
        self.edge_thickness = []
        self.load_model()

    def load_model(self) -> bool:
        if not self.obj_path or not os.path.exists(self.obj_path):
            print(f"[3D CAD] No se encontró el archivo OBJ en {self.obj_path}. Se usará modelo paramétrico.")
            return False

        try:
            t0 = time.time()
            verts = []
            edges = set()
            
            with open(self.obj_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if line.startswith('v '):
                        parts = line.strip().split()
                        verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    elif line.startswith('f '):
                        parts = line.strip().split()[1:]
                        idx = [int(p.split('/')[0]) - 1 for p in parts]
                        for i in range(len(idx)):
                            i1, i2 = idx[i], idx[(i + 1) % len(idx)]
                            if i1 != i2:
                                edges.add((min(i1, i2), max(i1, i2)))

            verts = np.array(verts, dtype=np.float32)
            if len(verts) == 0 or len(edges) == 0:
                return False

            # Centro del modelo CAD (en mm)
            c_x, c_y, c_z = 85.5, -45.0, 0.0
            
            # Transformación a marco del cuerpo FRD (Forward, Right, Down en metros):
            # CAD X -> +Forward
            # CAD Z -> +Right
            # CAD -Y -> +Down
            verts_body = np.zeros_like(verts)
            verts_body[:, 0] = (verts[:, 0] - c_x) * 0.001
            verts_body[:, 1] = (verts[:, 2] - c_z) * 0.001
            verts_body[:, 2] = -(verts[:, 1] - c_y) * 0.001

            edge_arr = np.array(list(edges), dtype=np.int32)
            v1_all = verts_body[edge_arr[:, 0]]
            v2_all = verts_body[edge_arr[:, 1]]
            lengths = np.linalg.norm(v2_all - v1_all, axis=1)

            # Filtrar aristas estructurales (longitud >= 5 mm)
            mask = (lengths >= 0.005)
            v1_filt = v1_all[mask]
            v2_filt = v2_all[mask]

            # Decimar uniformemente para mantener alta tasa de cuadros (60 FPS)
            if len(v1_filt) > self.max_edges:
                step = len(v1_filt) // self.max_edges
                v1_filt = v1_filt[::step]
                v2_filt = v2_filt[::step]

            self.v1_body = v1_filt
            self.v2_body = v2_filt
            
            # Asignar paleta de colores Bumblebee (Amarillo / Negro Carbón / Naranja)
            mid_z = (self.v1_body[:, 2] + self.v2_body[:, 2]) * 0.5
            mid_x = (self.v1_body[:, 0] + self.v2_body[:, 0]) * 0.5
            
            self.edge_colors = []
            for i in range(len(self.v1_body)):
                # Parte superior / cubierta en Amarillo Bumblebee
                if mid_z[i] < -0.02:
                    self.edge_colors.append((0, 215, 255)) # Amarillo oro
                elif mid_z[i] > 0.05:
                    # Patas de aterrizaje en Gris grafito / azul
                    self.edge_colors.append((220, 160, 60))
                else:
                    # Chasis y brazos de fibra de carbono
                    self.edge_colors.append((190, 190, 190))

            self.is_loaded = True
            print(f"[3D CAD] Modelo BUMBLEBEE.obj cargado con éxito ({len(self.v1_body)} aristas activas) en {time.time()-t0:.2f}s")
            return True

        except Exception as e:
            print(f"[3D CAD] Error al cargar BUMBLEBEE.obj: {e}")
            return False

    def render(self, canvas: np.ndarray, drone_pos_ned: np.ndarray,
               R_body_to_world: np.ndarray, R_w2c: np.ndarray,
               p_cam_world: np.ndarray, project_fn=None, f_px: float = 500.0) -> bool:
        """
        Renderiza el modelo CAD del dron proyectado en la cámara 3D de forma ultra-rápida (vectorizada).
        """
        if not self.is_loaded or len(self.v1_body) == 0:
            return False

        h, w = canvas.shape[:2]
        cx, cy = w // 2, h // 2

        # 1. Rotar y trasladar todos los vértices del modelo al mundo NED (Vectorizado)
        v1_world = drone_pos_ned + (R_body_to_world @ self.v1_body.T).T
        v2_world = drone_pos_ned + (R_body_to_world @ self.v2_body.T).T

        # 2. Transformar al marco de la cámara virtual 3D
        p1_cam = (R_w2c @ (v1_world - p_cam_world).T).T
        p2_cam = (R_w2c @ (v2_world - p_cam_world).T).T

        # Filtrar aristas delante del plano focal de la cámara (Z_cam > 0.1 m)
        z1 = p1_cam[:, 2]
        z2 = p2_cam[:, 2]
        valid = (z1 > 0.1) & (z2 > 0.1)
        if not np.any(valid):
            return True

        indices = np.where(valid)[0]
        u1 = (cx + (f_px * p1_cam[indices, 0] / z1[indices])).astype(np.int32)
        v1 = (cy + (f_px * p1_cam[indices, 1] / z1[indices])).astype(np.int32)
        u2 = (cx + (f_px * p2_cam[indices, 0] / z2[indices])).astype(np.int32)
        v2 = (cy + (f_px * p2_cam[indices, 1] / z2[indices])).astype(np.int32)

        # 3. Dibujar aristas visibles en canvas
        in_bounds = ((u1 >= -100) & (u1 < w + 100) & (v1 >= -100) & (v1 < h + 100)) | \
                    ((u2 >= -100) & (u2 < w + 100) & (v2 >= -100) & (v2 < h + 100))
        
        valid_idxs = indices[in_bounds]
        u1_b = u1[in_bounds]
        v1_b = v1[in_bounds]
        u2_b = u2[in_bounds]
        v2_b = v2[in_bounds]

        for k in range(len(valid_idxs)):
            orig_idx = valid_idxs[k]
            cv2.line(canvas, (u1_b[k], v1_b[k]), (u2_b[k], v2_b[k]), self.edge_colors[orig_idx], 1, cv2.LINE_AA)

        return True
