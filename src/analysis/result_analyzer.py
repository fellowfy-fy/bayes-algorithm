import numpy as np
import pandas as pd
from typing import Dict
from datetime import datetime
from scipy import stats
import logging
from config.monitoring_config import MonitoringConfig

logger = logging.getLogger(__name__)

class ResultAnalyzer:
    """Класс для анализа результатов мониторинга"""
    def __init__(self, config: Dict):
        self.config = config
        self.frequency_history = []
        self.environmental_factors = []
        self.damage_indicators = {}
        self.thresholds = config.get('thresholds', {})
        
    def analyze_frequencies(self, new_frequencies: np.ndarray,
                          environmental_data: Dict) -> Dict:
        """Анализ изменений частот с учетом внешних факторов"""
        self.frequency_history.append({
            'timestamp': datetime.now(),
            'frequencies': new_frequencies,
            'environmental': environmental_data
        })
        
        if len(self.frequency_history) <= self.config.get('min_history_length', 10):
            return {'current_frequencies': new_frequencies}
            
        trend_window = self.config.get('trend_window', 24)
        freq_array = np.array([
            item['frequencies'] for item in self.frequency_history[-trend_window:]
        ])
        
        trend_analysis = {
            'mean': np.mean(freq_array, axis=0),
            'std': np.std(freq_array, axis=0),
            'trend': np.polyfit(range(len(freq_array)), freq_array, deg=1)[0]
        }
        
        env_data = pd.DataFrame([
            item['environmental'] 
            for item in self.frequency_history[-trend_window:]
        ])
        
        correlations = {}
        for factor in env_data.columns:
            correlations[factor] = np.corrcoef(
                env_data[factor],
                freq_array[:, 0]
            )[0, 1]
            
        return {
            'trend_analysis': trend_analysis,
            'environmental_correlations': correlations,
            'current_frequencies': new_frequencies
        }

    def detect_damage(self, frequency_analysis: Dict,
                     structural_response: np.ndarray) -> Dict:
        """Выявление возможных повреждений"""
        damage_indicators = {}
        
        if 'trend_analysis' in frequency_analysis:
            if isinstance(frequency_analysis['trend_analysis']['trend'], np.ndarray):
                freq_change = np.max(np.abs(frequency_analysis['trend_analysis']['trend']))
            else:
                freq_change = abs(frequency_analysis['trend_analysis']['trend'])
                
            damage_indicators['frequency_change'] = {
                'value': float(freq_change),
                'exceeds_threshold': freq_change > self.thresholds.get('frequency_change', float('inf'))
            }
        
        if structural_response is not None and len(structural_response) > 0:
            if structural_response.ndim > 1:
                max_amplitude = np.max(np.abs(structural_response))
                rms = np.sqrt(np.mean(np.sum(structural_response**2, axis=1)))
                flattened_response = structural_response.flatten()
                kurtosis = stats.kurtosis(flattened_response) if len(flattened_response) > 3 else 0
            else:
                max_amplitude = np.max(np.abs(structural_response))
                rms = np.sqrt(np.mean(structural_response**2))
                kurtosis = stats.kurtosis(structural_response) if len(structural_response) > 3 else 0
            
            response_features = {
                'max_amplitude': max_amplitude,
                'rms': rms,
                'kurtosis': kurtosis
            }
            
            for feature, value in response_features.items():
                damage_indicators[feature] = {
                    'value': float(value),
                    'exceeds_threshold': value > self.thresholds.get(feature, float('inf'))
                }
        
        return damage_indicators