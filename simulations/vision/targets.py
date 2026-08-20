"""
Generador de texturas de objetivos visuales (Marcador ArUco / Código QR / Plataforma SkyDock).
Crea imágenes de alta resolución que representan la estación de aterrizaje en el suelo.
"""
import cv2
import numpy as np
from typing import Tuple, Optional

class VisualTargetGenerator:
    def __init__(self, pad_size_m: float = 0.80, resolution_px_per_m: int = 1500):
        """
        :param pad_size_m: Tamaño físico de la plataforma cuadrada en metros (ej. 0.80 m).
        :param resolution_px_per_m: Resolución de renderizado en píxeles por metro.
        """
        self.pad_size_m = pad_size_m
        self.resolution_px_per_m = resolution_px_per_m
        self.img_dim = int(pad_size_m * resolution_px_per_m)
        self.dict_type = cv2.aruco.DICT_4X4_50

    def generate_aruco_marker(self, marker_id: int = 0, marker_size_px: int = 600) -> np.ndarray:
        """Genera un marcador ArUco binario nítido."""
        if hasattr(cv2.aruco, 'getPredefinedDictionary'):
            dictionary = cv2.aruco.getPredefinedDictionary(self.dict_type)
            marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size_px)
        else:
            dictionary = cv2.aruco.Dictionary_get(self.dict_type)
            marker_img = cv2.aruco.drawMarker(dictionary, marker_id, marker_size_px)
        return marker_img

    def generate_qr_code(self, data: str = "SKYDOCK_PORT_01", qr_size_px: int = 600) -> np.ndarray:
        """Genera un código QR nítido usando patrones QR estándar o OpenCV."""
        # Creamos una matriz QR sintética de alto contraste con patrones de alineación
        qr_img = np.ones((qr_size_px, qr_size_px), dtype=np.uint8) * 255
        
        # Grid 21x21 (QR Version 1)
        grid_size = 21
        cell_size = qr_size_px // (grid_size + 4) # con quiet zone
        offset = (qr_size_px - (grid_size * cell_size)) // 2
        
        def draw_finder_pattern(row, col):
            top_left = (offset + col * cell_size, offset + row * cell_size)
            # 7x7 outer black
            cv2.rectangle(qr_img, top_left, 
                          (top_left[0] + 7 * cell_size, top_left[1] + 7 * cell_size), 0, -1)
            # 5x5 inner white
            cv2.rectangle(qr_img, (top_left[0] + cell_size, top_left[1] + cell_size),
                          (top_left[0] + 6 * cell_size, top_left[1] + 6 * cell_size), 255, -1)
            # 3x3 center black
            cv2.rectangle(qr_img, (top_left[0] + 2 * cell_size, top_left[1] + 2 * cell_size),
                          (top_left[0] + 5 * cell_size, top_left[1] + 5 * cell_size), 0, -1)

        # 3 Finder patterns
        draw_finder_pattern(0, 0)
        draw_finder_pattern(0, grid_size - 7)
        draw_finder_pattern(grid_size - 7, 0)
        
        # Timing patterns
        for i in range(8, grid_size - 8):
            color = 0 if (i % 2 == 0) else 255
            cv2.rectangle(qr_img, (offset + 6 * cell_size, offset + i * cell_size),
                          (offset + 7 * cell_size, offset + (i + 1) * cell_size), color, -1)
            cv2.rectangle(qr_img, (offset + i * cell_size, offset + 6 * cell_size),
                          (offset + (i + 1) * cell_size, offset + 7 * cell_size), color, -1)
            
        # Payload pseudorandom bit pattern basada en el hash del texto
        rng = np.random.RandomState(abs(hash(data)) % (2**31))
        for r in range(grid_size):
            for c in range(grid_size):
                if (r < 8 and c < 8) or (r < 8 and c >= grid_size - 8) or (r >= grid_size - 8 and c < 8):
                    continue # Zona protegida de finders
                if r == 6 or c == 6:
                    continue # Timing line
                if rng.rand() > 0.5:
                    cv2.rectangle(qr_img, (offset + c * cell_size, offset + r * cell_size),
                                  (offset + (c + 1) * cell_size, offset + (r + 1) * cell_size), 0, -1)
        return qr_img

    def create_skydock_landing_pad(self, target_type: str = "aruco", marker_id: int = 0,
                                   marker_physical_size_m: float = 0.20,
                                   inner_marker_size_m: Optional[float] = 0.05) -> Tuple[np.ndarray, dict]:
        """
        Crea la textura completa de la plataforma SkyDock en alta resolución.
        
        :return: (imagen_uint8_mono, info_dict)
        """
        pad = np.ones((self.img_dim, self.img_dim), dtype=np.uint8) * 190 # Fondo gris metálico/concreto
        
        # Borde exterior negro y franja de seguridad amarilla/negra (en mono es contraste)
        margin_px = int(0.04 * self.resolution_px_per_m)
        cv2.rectangle(pad, (margin_px, margin_px), 
                      (self.img_dim - margin_px, self.img_dim - margin_px), 30, int(0.015 * self.resolution_px_per_m))
        
        # Círculo de helipad / zona de aterrizaje
        center_px = self.img_dim // 2
        radius_outer_px = int((self.pad_size_m * 0.42) * self.resolution_px_per_m)
        cv2.circle(pad, (center_px, center_px), radius_outer_px, 255, int(0.012 * self.resolution_px_per_m))
        
        # Líneas de alineación / Crosshairs
        line_len_px = int((self.pad_size_m * 0.45) * self.resolution_px_per_m)
        cv2.line(pad, (center_px - line_len_px, center_px), (center_px + line_len_px, center_px), 230, int(0.005 * self.resolution_px_per_m))
        cv2.line(pad, (center_px, center_px - line_len_px), (center_px, center_px + line_len_px), 230, int(0.005 * self.resolution_px_per_m))

        # Cuadrado blanco de alto contraste en el centro para el marcador
        main_marker_px = int(marker_physical_size_m * self.resolution_px_per_m)
        bg_box_px = int((marker_physical_size_m + 0.04) * self.resolution_px_per_m)
        p1 = center_px - bg_box_px // 2
        p2 = center_px + bg_box_px // 2
        cv2.rectangle(pad, (p1, p1), (p2, p2), 255, -1)
        cv2.rectangle(pad, (p1, p1), (p2, p2), 0, int(0.004 * self.resolution_px_per_m))

        if target_type.lower() == "aruco":
            marker_img = self.generate_aruco_marker(marker_id=marker_id, marker_size_px=main_marker_px)
            m_top = center_px - main_marker_px // 2
            m_left = center_px - main_marker_px // 2
            pad[m_top:m_top + main_marker_px, m_left:m_left + main_marker_px] = marker_img

        elif target_type.lower() == "qr":
            qr_img = self.generate_qr_code(data=f"SKYDOCK_LANDING_PAD_ID_{marker_id}", qr_size_px=main_marker_px)
            m_top = center_px - main_marker_px // 2
            m_left = center_px - main_marker_px // 2
            pad[m_top:m_top + main_marker_px, m_left:m_left + main_marker_px] = qr_img

        info = {
            "pad_size_m": self.pad_size_m,
            "resolution_px_per_m": self.resolution_px_per_m,
            "img_dim_px": self.img_dim,
            "center_px": (center_px, center_px),
            "target_type": target_type,
            "marker_id": marker_id,
            "marker_physical_size_m": marker_physical_size_m,
            "inner_marker_size_m": inner_marker_size_m
        }
        return pad, info
