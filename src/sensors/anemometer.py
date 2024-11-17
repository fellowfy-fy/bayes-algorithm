from datetime import datetime
from typing import Tuple
import numpy as np
from .base import Sensor

class Anemometer(Sensor):
    """Класс для анемометра"""
    def __init__(self, sensor_id: str, location: Tuple[float, float, float]):
        super().__init__(sensor_id, location)
        self.wind_direction = 0.0
        
    def get_reading(self) -> np.ndarray:
        """Получение данных о скорости и направлении ветра"""
        speed = abs(np.random.normal(5, 2))
        direction = (self.wind_direction + np.random.normal(0, 10)) % 360
        self.wind_direction = direction
        self.last_reading_time = datetime.now()
        return np.array([speed, direction])