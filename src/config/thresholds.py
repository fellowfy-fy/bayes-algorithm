from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class MonitoringThresholds:
    """Пороговые значения для мониторинга"""
    frequency_change: float
    displacement: float
    acceleration: float
    wind_speed: float
    temperature_range: Tuple[float, float]
    humidity_range: Tuple[float, float]
    
    def to_dict(self) -> Dict:
        """Преобразование в словарь для сериализации"""
        return {
            'frequency_change': self.frequency_change,
            'displacement': self.displacement,
            'acceleration': self.acceleration,
            'wind_speed': self.wind_speed,
            'temperature_range': self.temperature_range,
            'humidity_range': self.humidity_range
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MonitoringThresholds':
        """Создание объекта из словаря"""
        return cls(**data)