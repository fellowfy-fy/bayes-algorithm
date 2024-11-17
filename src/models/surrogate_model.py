# models/surrogate_model.py

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class SurrogateModel:
    """Класс для создания и обучения суррогатной модели"""
    def __init__(self, input_dim: int):
        # Создаем более гибкое ядро
        kernel = ConstantKernel(1.0) * RBF(
            length_scale=[1.0] * input_dim,
            length_scale_bounds=[(1e-5, 1e5)] * input_dim
        ) + WhiteKernel(noise_level=0.1)
        
        self.gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            normalize_y=True,  # Нормализация выходных данных
            random_state=42
        )
        
        self.input_dim = input_dim
        self.is_trained = False
        self.training_points = []
        self.training_values = []
        self.scaler = StandardScaler()
        
    def add_training_point(self, input_params: np.ndarray, output_values: np.ndarray):
        """Добавление точки для обучения"""
        input_params = np.asarray(input_params).reshape(1, -1)
        output_values = np.asarray(output_values).reshape(1, -1)
        
        self.training_points.append(input_params)
        self.training_values.append(output_values)
        
    def train(self):
        """Обучение суррогатной модели"""
        if len(self.training_points) < 2:
            raise ValueError("Необходимо минимум 2 точки для обучения")
            
        X = np.vstack(self.training_points)
        y = np.vstack(self.training_values)
        
        # Нормализация входных данных
        X_scaled = self.scaler.fit_transform(X)
        
        # Обучение модели
        try:
            self.gpr.fit(X_scaled, y)
            self.is_trained = True
            logger.info("Surrogate model trained successfully")
            logger.info(f"Final kernel parameters: {self.gpr.kernel_}")
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise
        
    def predict(self, input_params: np.ndarray, 
                return_std: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Предсказание с помощью суррогатной модели"""
        if not self.is_trained:
            raise ValueError("Модель должна быть сначала обучена")
        
        input_params = np.asarray(input_params)
        if input_params.ndim == 1:
            input_params = input_params.reshape(1, -1)
            
        # Нормализация входных данных
        X_scaled = self.scaler.transform(input_params)
        
        try:
            if return_std:
                mean, std = self.gpr.predict(X_scaled, return_std=True)
                return mean.reshape(-1), std
            else:
                mean = self.gpr.predict(X_scaled)
                return mean.reshape(-1), None
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise