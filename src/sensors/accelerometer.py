from datetime import datetime
from typing import Tuple
import numpy as np
from .base import Sensor

class Accelerometer(Sensor):
    """Класс для акселерометра"""
    def __init__(self, sensor_id: str, location: Tuple[float, float, float]):
        super().__init__(sensor_id, location)
        self.axis_sensitivity = np.array([1.0, 1.0, 1.0])

    def get_reading(self) -> np.ndarray:
        """Получение данных с акселерометра"""
        t = datetime.now().timestamp()
        base_signal = np.sin(2 * np.pi * 1.5 * t) + np.sin(2 * np.pi * 2.5 * t)
        reading = (base_signal + np.random.normal(0, 0.1, 3)) * self.axis_sensitivity
        self.last_reading_time = datetime.now()
        return reading