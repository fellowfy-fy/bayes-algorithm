from datetime import datetime
import numpy as np
from typing import Tuple
from sklearn.cluster import KMeans
from scipy.signal import find_peaks
from .data_collector import DataCollector

class InitialTesting:
    """Класс для проведения начального тестирования (AVT)"""
    def __init__(self, data_collector: DataCollector):
        self.data_collector = data_collector
        self.f1 = None  # первая частота (крутильная)
        self.f2 = None  # вторая частота (поступательная)

    def perform_vibration_test(self, test_duration: float = 3600) -> Tuple[float, float]:
        """
        Проведение вибрационного теста и определение основных частот
        
        Args:
            test_duration: продолжительность теста в секундах
        
        Returns:
            Tuple[float, float]: кортеж (f1, f2) с основными частотами
        """
        # Собираем данные в течение указанного времени
        data = self.data_collector.collect_data(test_duration)
        
        # Список для хранения найденных частот
        all_frequencies = []
        
        # Обработка данных для каждого акселерометра
        for i in range(self.data_collector.config.num_accelerometers):
            for axis in ['x', 'y', 'z']:
                column_name = f'acc_{i}_{axis}'
                if column_name in data.columns:
                    signal = data[column_name].values
                    
                    # Выполняем FFT
                    fft = np.fft.fft(signal)
                    freq = np.fft.fftfreq(len(signal), 
                                        1/self.data_collector.config.sampling_rate)
                    
                    # Находим пики в спектре
                    magnitude = np.abs(fft)[1:len(freq)//2]
                    frequencies = freq[1:len(freq)//2]
                    peaks, _ = find_peaks(magnitude, height=np.max(magnitude)*0.1)
                    
                    if len(peaks) > 0:
                        peak_frequencies = frequencies[peaks]
                        all_frequencies.extend(peak_frequencies)

        if len(all_frequencies) < 2:
            raise ValueError("Недостаточно данных для определения двух основных частот")

        # Кластеризуем частоты
        freq_array = np.array(all_frequencies).reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(freq_array)
        
        # Получаем центры кластеров и сортируем их
        cluster_centers = kmeans.cluster_centers_.flatten()
        sorted_centers = np.sort(cluster_centers)
        
        self.f1, self.f2 = sorted_centers[0], sorted_centers[1]
        return self.f1, self.f2

    def generate_test_report(self) -> dict:
        """Генерация отчета о проведенном тестировании"""
        if self.f1 is None or self.f2 is None:
            raise ValueError("Необходимо сначала провести вибрационный тест")
            
        return {
            'test_time': datetime.now(),
            'f1_torsional': self.f1,
            'f2_translational': self.f2,
            'test_duration': 3600,
            'number_of_sensors': self.data_collector.config.num_accelerometers,
            'sampling_rate': self.data_collector.config.sampling_rate
        }