from typing import Tuple
import numpy as np
from datetime import datetime

class Sensor:
    """Базовый класс для всех сенсоров"""
    def __init__(self, sensor_id: str, location: Tuple[float, float, float]):
        self.sensor_id = sensor_id
        self.location = location
        self.is_active = True
        self.last_reading_time = None

    def get_reading(self) -> np.ndarray:
        raise NotImplementedError("Метод должен быть реализован в подклассах")
