from datetime import datetime
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class DecisionMaker:
    """Класс для принятия решений на основе анализа"""
    def __init__(self, config: Dict):
        self.config = config
        self.action_history = []
        self.current_status = 'normal'
        
    def evaluate_condition(self, damage_indicators: Dict,
                         frequency_analysis: Dict) -> Dict:
        """Оценка состояния и принятие решений"""
        severity_score = 0
        warnings_list = []
        
        for indicator, data in damage_indicators.items():
            if data['exceeds_threshold']:
                severity_score += self.config['indicator_weights'].get(indicator, 1.0)
                warnings_list.append(f"Превышен порог {indicator}")
        
        if 'trend_analysis' in frequency_analysis:
            trend = frequency_analysis['trend_analysis']['trend']
            if abs(trend) > self.config['critical_trend']:
                severity_score += self.config['trend_weight']
                warnings_list.append("Критическое изменение частот")
        
        actions = self._determine_actions(severity_score, warnings_list)
        
        decision_result = {
            'timestamp': datetime.now(),
            'severity_score': severity_score,
            'warnings': warnings_list,
            'recommended_actions': actions,
            'status': self._determine_status(severity_score)
        }
        
        self.action_history.append(decision_result)
        self.current_status = decision_result['status']
        
        return decision_result
    
    def _determine_status(self, severity_score: float) -> str:
        """Определение статуса на основе оценки серьезности"""
        if severity_score <= self.config['normal_threshold']:
            return 'normal'
        elif severity_score <= self.config['warning_threshold']:
            return 'warning'
        elif severity_score <= self.config['alert_threshold']:
            return 'alert'
        else:
            return 'critical'
    
    def _determine_actions(self, severity_score: float,
                         warnings: List[str]) -> List[str]:
        """Определение необходимых действий"""
        actions = []
        
        if severity_score > self.config['alert_threshold']:
            actions.extend([
                "Немедленная инспекция конструкции",
                "Уведомление ответственных лиц",
                "Подготовка к возможной эвакуации"
            ])
        elif severity_score > self.config['warning_threshold']:
            actions.extend([
                "Внеплановая проверка",
                "Увеличение частоты мониторинга",
                "Подготовка отчета о состоянии"
            ])
        else:
            actions.append("Продолжение штатного мониторинга")
            
        return actions