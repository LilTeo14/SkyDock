"""
Manejador centralizado de logs y gestión de ensayos para la simulación SkyDock.
Permite iniciar/detener grabaciones bajo demanda con numeración consecutiva
(ej. ensayo_001_20260820_013500.log y .csv) y realizar vuelos libres sin grabar.
"""
import sys
import os
import re
from pathlib import Path
from datetime import datetime
import threading
from typing import Optional, Tuple

class SkyDockLogger:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.logs_dir = Path(__file__).resolve().parent.parent / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.is_recording = False
        self.current_trial_num = 0
        self.current_trial_id = ""
        self.log_file = None
        self.log_path: Optional[Path] = None

    @classmethod
    def get_logger(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def get_next_trial_number(self) -> int:
        """Determina el siguiente número de prueba correlativo analizando los archivos existentes."""
        max_num = 0
        pattern = re.compile(r'ensayo_(\d+)_')
        if self.logs_dir.exists():
            for file in self.logs_dir.iterdir():
                match = pattern.search(file.name)
                if match:
                    try:
                        num = int(match.group(1))
                        if num > max_num:
                            max_num = num
                    except ValueError:
                        pass
        return max_num + 1

    def start_trial(self) -> Tuple[int, str]:
        """
        Inicia la grabación formal de un nuevo ensayo.
        :return: (trial_number, trial_id_str)
        """
        with self._lock:
            if self.is_recording:
                self.stop_trial()

            self.current_trial_num = self.get_next_trial_number()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.current_trial_id = f"ensayo_{self.current_trial_num:03d}_{timestamp}"
            
            self.log_path = self.logs_dir / f"{self.current_trial_id}.log"
            self.log_file = open(str(self.log_path), mode='w', encoding='utf-8', buffering=1)
            self.is_recording = True
            
            self.log(f"================================================================")
            self.log(f" >>> INICIANDO GRABACIÓN: ENSAYO #{self.current_trial_num:03d} ({timestamp}) <<<")
            self.log(f"================================================================")
            return self.current_trial_num, self.current_trial_id

    def stop_trial(self) -> Optional[int]:
        """Detiene y guarda la grabación del ensayo actual."""
        with self._lock:
            if not self.is_recording:
                return None

            num = self.current_trial_num
            self.log(f"================================================================")
            self.log(f" >>> FINALIZADO Y GUARDADO: ENSAYO #{num:03d} <<<")
            self.log(f" Archivo Log: {self.log_path.name if self.log_path else ''}")
            self.log(f"================================================================")

            if self.log_file and not self.log_file.closed:
                self.log_file.close()
                self.log_file = None

            self.is_recording = False
            self.log_path = None
            return num

    def log(self, message: str):
        """Imprime en consola y registra en el archivo del ensayo si está grabando."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        formatted = f"[{timestamp}] {message}"
        
        # Siempre imprimir en terminal
        print(message)
        
        # Escribir a archivo si está en modo grabación
        if self.is_recording and self.log_file and not self.log_file.closed:
            try:
                self.log_file.write(formatted + "\n")
            except Exception:
                pass

def log_event(message: str):
    """Función global de logging de eventos."""
    logger = SkyDockLogger.get_logger()
    logger.log(message)
