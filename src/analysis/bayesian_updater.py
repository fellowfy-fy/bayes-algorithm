import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
from datetime import datetime
from scipy.stats import multivariate_normal
import logging
from models.surrogate_model import SurrogateModel

logger = logging.getLogger(__name__)

@dataclass
class BayesianParameters:
    """Параметры для байесовского обновления"""
    prior_mean: np.ndarray
    prior_cov: np.ndarray
    likelihood_std: float
    confidence_level: float = 0.95

class BayesianUpdater:
    """Класс для байесовского обновления модели"""
    def __init__(self, initial_params: BayesianParameters,
                 surrogate_model: SurrogateModel):
        self.params = initial_params
        self.surrogate_model = surrogate_model
        self.update_history = []
        
    def compute_likelihood(self, measurement: np.ndarray, 
                         predicted: np.ndarray) -> float:
        """Вычисление функции правдоподобия"""
        try:
            measurement = np.array(measurement).reshape(-1)
            predicted = np.array(predicted).reshape(-1)
            
            if len(measurement) != len(predicted):
                raise ValueError(f"Dimension mismatch: measurement has length {len(measurement)}, but predicted has length {len(predicted)}")
            
            cov_matrix = self.params.likelihood_std**2 * np.eye(len(measurement))
            
            return multivariate_normal.pdf(
                measurement,
                mean=predicted,
                cov=cov_matrix
            )
        except Exception as e:
            logger.error(f"Error computing likelihood: {str(e)}")
            raise
    
    def update(self, measurement: np.ndarray) -> Dict:
        """Выполнение байесовского обновления"""
        try:
            measurement = np.array(measurement).reshape(-1)
            measurement_dim = len(measurement)
            prior_dim = len(self.params.prior_mean)
            
            predicted_mean, predicted_std = self.surrogate_model.predict(
                self.params.prior_mean, 
                return_std=True
            )
            
            predicted_mean = predicted_mean[:measurement_dim]
            likelihood = self.compute_likelihood(measurement, predicted_mean)
            measurement_cov = self.params.likelihood_std**2 * np.eye(measurement_dim)
            
            # Калмановское усиление
            K = np.zeros((prior_dim, measurement_dim))
            for i in range(prior_dim):
                for j in range(measurement_dim):
                    K[i,j] = self.params.prior_cov[i,i] / (self.params.prior_cov[i,i] + measurement_cov[j,j])
            
            innovation = measurement - predicted_mean
            update = np.dot(K, innovation)
            posterior_mean = self.params.prior_mean + update
            
            posterior_cov = (
                np.eye(prior_dim) - np.dot(K, np.ones((measurement_dim, prior_dim)))
            ) @ self.params.prior_cov
            
            update_result = {
                'timestamp': datetime.now(),
                'prior_mean': self.params.prior_mean.copy(),
                'posterior_mean': posterior_mean,
                'likelihood': likelihood,
                'measurement': measurement,
                'predicted': predicted_mean
            }
            
            self.params.prior_mean = posterior_mean
            self.params.prior_cov = posterior_cov
            self.update_history.append(update_result)
            
            return update_result
            
        except Exception as e:
            logger.error(f"Error in Bayesian update: {str(e)}")
            raise