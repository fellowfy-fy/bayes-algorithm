import pandas as pd
from datetime import datetime
from typing import List
from config.monitoring_config import MonitoringConfig
from sensors.accelerometer import Accelerometer
from sensors.anemometer import Anemometer

class DataCollector:
    """Класс для сбора и предварительной обработки данных"""
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.accelerometers: List[Accelerometer] = []
        self.anemometers: List[Anemometer] = []
        self.initialize_sensors()

    def initialize_sensors(self):
        """Инициализация сенсоров"""
        # Создаем акселерометры
        for i in range(self.config.num_accelerometers):
            location = (0, 0, 10 * (i + 1))  # Пример расположения
            self.accelerometers.append(
                Accelerometer(f"ACC_{i}", location)
            )
        
        # Создаем анемометр
        self.anemometers.append(
            Anemometer("ANEM_1", (0, 0, 30))  # Размещаем на верху здания
        )

    def collect_data(self, duration: float) -> pd.DataFrame:
        """
        Сбор данных со всех сенсоров за указанный период
        
        Args:
            duration: продолжительность сбора данных в секундах
        
        Returns:
            DataFrame с собранными данными
        """
        num_samples = int(duration * self.config.sampling_rate)
        data = []
        
        for _ in range(num_samples):
            timestamp = datetime.now()
            sample = {'timestamp': timestamp}
            
            # Сбор данных с акселерометров
            for i, acc in enumerate(self.accelerometers):
                reading = acc.get_reading()
                sample.update({
                    f'acc_{i}_x': reading[0],
                    f'acc_{i}_y': reading[1],
                    f'acc_{i}_z': reading[2]
                })
            
            # Сбор данных с анемометра
            wind_data = self.anemometers[0].get_reading()
            sample.update({
                'wind_speed': wind_data[0],
                'wind_direction': wind_data[1]
            })
            
            data.append(sample)
        
        return pd.DataFrame(data)