from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import logging

from config.monitoring_config import MonitoringConfig
from config.thresholds import MonitoringThresholds
from core.data_collector import DataCollector
from core.initial_testing import InitialTesting
from models.fe_model import FEModel
from models.calibrator import ModelCalibrator
from models.surrogate_model import SurrogateModel
from analysis.bayesian_updater import BayesianParameters, BayesianUpdater
from analysis.result_analyzer import ResultAnalyzer
from analysis.decision_maker import DecisionMaker
from utils.data_storage import DataStorage
from utils.visualization import MonitoringVisualizer
from utils.logger import setup_logger

logger = logging.getLogger(__name__)

def prepare_data_for_storage(measurement_data, frequency_analysis, damage_indicators, decision):
    """Подготовка данных для сохранения"""
    return {
        'timestamp': datetime.now().isoformat(),
        'measurements': {
            'accelerometer': measurement_data[['acc_0_x', 'acc_0_y', 'acc_0_z']].to_dict('records')[0],
            'wind': {
                'speed': float(measurement_data['wind_speed'].mean()),
                'direction': float(measurement_data['wind_direction'].mean())
            }
        },
        'analysis': {
            'frequencies': frequency_analysis.get('current_frequencies', []).tolist(),
            'damage_indicators': {
                k: float(v['value']) if isinstance(v, dict) else float(v)
                for k, v in damage_indicators.items()
            }
        },
        'decision': {
            'status': decision['status'],
            'severity_score': float(decision['severity_score']),
            'warnings': decision['warnings'],
            'recommended_actions': decision['recommended_actions']
        }
    }

def run_monitoring():
    """Пример использования системы мониторинга"""
    try:
        # 1. Настройка базовой конфигурации
        config = MonitoringConfig(
            sampling_rate=100.0,  # Гц
            measurement_interval=10.0,  # 1 час
            num_accelerometers=3,
            num_anemometers=1
        )
        
        # 2. Настройка пороговых значений
        thresholds = MonitoringThresholds(
            frequency_change=0.1,
            displacement=0.05,
            acceleration=0.2,
            wind_speed=30.0,
            temperature_range=(-20.0, 40.0),
            humidity_range=(20.0, 80.0)
        )
        
        # 3. Инициализация хранилища данных и логирования
        storage_path = Path("monitoring_data")
        logs_path = Path("logs")
        data_storage = DataStorage(storage_path)
        setup_logger(logs_path)
        visualizer = MonitoringVisualizer(storage_path / "plots")
        
        # 4. Создание сборщика данных и проведение начального тестирования
        print("Инициализация системы мониторинга...")
        data_collector = DataCollector(config)
        initial_testing = InitialTesting(data_collector)
        
        print("Проведение начального тестирования...")
        f1, f2 = initial_testing.perform_vibration_test(test_duration=3600)
        test_report = initial_testing.generate_test_report()
        print(f"Определены собственные частоты:")
        print(f"f1 (крутильная): {f1:.2f} Гц")
        print(f"f2 (поступательная): {f2:.2f} Гц")
        
        # 5. Настройка и калибровка моделей
        print("\nНастройка моделей...")
        building_params = {
            'n_degrees_of_freedom': 5,
            'floor_mass': 1000.0,  # кг
            'story_stiffness': 2e6,  # Н/м
            'isolator_stiffness': 1e6,  # Н/м
            'rayleigh_alpha': 0.1,
            'rayleigh_beta': 0.1,
            'sd_friction': 0.05  # начальное значение коэффициента трения
        }
        
        print("\nСоздание и калибровка FE модели...")
        fe_model = FEModel(building_params)
        calibrator = ModelCalibrator(fe_model, (f1, f2))
        
        # Калибровка модели
        calibration_result = calibrator.calibrate(
            initial_guess=np.array([1e6, 0.05]),  # [initial_stiffness, initial_friction]
            bounds=[(5e5, 5e6), (0.01, 0.1)]  # диапазоны для параметров
        )
        
        print("Результаты калибровки:")
        print(f"LRB жесткость: {calibration_result['lrb_stiffness']:.2e} Н/м")
        print(f"SD трение: {calibration_result['sd_friction']:.3f}")
        
        # 6. Обучение суррогатной модели
        print("\nОбучение суррогатной модели...")
        surrogate_model = SurrogateModel(input_dim=2)
        
        # Создаем сетку параметров для обучения
        stiffness_range = np.logspace(5, 7, 10)  # от 1e5 до 1e7
        friction_range = np.linspace(0.01, 0.1, 10)
        
        # Генерация обучающих данных
        for stiffness in stiffness_range:
            for friction in friction_range:
                # Обновляем параметры FE модели
                fe_model.update_isolator_parameters(
                    lrb_stiffness=stiffness,
                    sd_friction=friction
                )
                
                # Вычисляем частоты и добавляем точку для обучения
                output_values = fe_model.compute_natural_frequencies()
                surrogate_model.add_training_point(
                    np.array([stiffness, friction]),
                    output_values
                )
        
        # Обучаем модель
        print("Обучение модели на", len(surrogate_model.training_points), "точках...")
        surrogate_model.train()
        
        # 7. Настройка байесовского обновления
        initial_bayes_params = BayesianParameters(
            prior_mean=np.array([1e6, 0.05]),
            prior_cov=np.eye(2),
            likelihood_std=0.1
        )
        
        bayesian_updater = BayesianUpdater(initial_bayes_params, surrogate_model)
        result_analyzer = ResultAnalyzer({'thresholds': thresholds.__dict__})
        
        # 8. Настройка принятия решений
        decision_config = {
            'normal_threshold': 1.0,
            'warning_threshold': 2.0,
            'alert_threshold': 3.0,
            'indicator_weights': {
                'frequency_change': 2.0,
                'displacement': 1.5,
                'acceleration': 1.0
            },
            'trend_weight': 1.5,
            'critical_trend': 0.2
        }
        
        decision_maker = DecisionMaker(decision_config)
        
        # 9. Основной цикл мониторинга
        print("\nЗапуск мониторинга...")
        monitoring_duration = timedelta(days=1)  # Пример: мониторинг в течение 1 дня
        start_time = datetime.now()
        current_time = start_time
        
        while current_time < start_time + monitoring_duration:
            try:
                # Сбор данных
                measurement_data = data_collector.collect_data(
                    duration=config.measurement_interval
                )
                
                # Байесовское обновление
                update_result = bayesian_updater.update(
                    measurement_data[['acc_0_x', 'acc_0_y', 'acc_0_z']].iloc[0].values
                )
                
                # Анализ результатов
                frequency_analysis = result_analyzer.analyze_frequencies(
                    new_frequencies=np.array([f1, f2]),
                    environmental_data={
                        'temperature': measurement_data['wind_speed'].mean(),
                        'wind_speed': measurement_data['wind_direction'].mean()
                    }
                )
                
                # Определение повреждений
                damage_indicators = result_analyzer.detect_damage(
                    frequency_analysis,
                    measurement_data[['acc_0_x', 'acc_0_y', 'acc_0_z']].values
                )
                
                # Принятие решений
                decision = decision_maker.evaluate_condition(
                    damage_indicators,
                    frequency_analysis
                )
                
                # Подготовка и сохранение данных
                storage_data = prepare_data_for_storage(
                    measurement_data,
                    frequency_analysis,
                    damage_indicators,
                    decision
                )
                
                # Сохранение результатов
                data_storage.save_json(
                    storage_data,
                    f"monitoring_{current_time.strftime('%Y%m%d_%H%M%S')}.json"
                )
                
                # Визуализация текущих результатов
                if len(measurement_data) > 0:
                    visualizer.plot_time_series(
                        measurement_data,
                        'acc_0_x',
                        'Acceleration X-axis'
                    )
                
                # Вывод текущего состояния
                print(f"\nВремя: {current_time}")
                print(f"Статус: {decision['status']}")
                if decision['warnings']:
                    print("Предупреждения:", decision['warnings'])
                if decision['recommended_actions']:
                    print("Рекомендуемые действия:", decision['recommended_actions'])
                
                current_time += timedelta(hours=1)
                
            except Exception as e:
                logger.error(f"Ошибка в цикле мониторинга: {str(e)}")
                raise
        
        print("\nМониторинг завершен. Результаты сохранены в папке 'monitoring_data'")
        
    except Exception as e:
        logger.error(f"Критическая ошибка в работе программы: {str(e)}")
        raise

if __name__ == "__main__":
    run_monitoring()