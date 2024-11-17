# structural_monitoring/models/fe_model.py

import numpy as np
from typing import Dict
from scipy.linalg import eigh
import logging

logger = logging.getLogger(__name__)

class FEModel:
    """Класс для работы с конечно-элементной моделью здания"""
    def __init__(self, building_params: Dict):
        self.params = building_params
        self.stiffness_matrix = None
        self.mass_matrix = None
        self.damping_matrix = None
        self.natural_frequencies = None
        self._initialize_matrices()
        
    def update_isolator_parameters(self, lrb_stiffness: float, sd_friction: float):
        """
        Обновление параметров изоляторов
        
        Args:
            lrb_stiffness: жесткость свинцово-резиновых опор
            sd_friction: коэффициент трения скользящих устройств
        """
        try:
            self.params['isolator_stiffness'] = lrb_stiffness
            self.params['sd_friction'] = sd_friction
            self._initialize_matrices()  # Пересчитываем матрицы с новыми параметрами
            
        except Exception as e:
            logger.error(f"Error updating isolator parameters: {str(e)}")
            raise
        
    def _initialize_matrices(self):
        """Инициализация матриц жесткости, массы и демпфирования"""
        try:
            n_dof = self.params['n_degrees_of_freedom']
            
            # Инициализация матрицы масс (диагональная)
            self.mass_matrix = np.diag(
                [self.params['floor_mass']] * n_dof
            )
            
            # Инициализация матрицы жесткости
            self.stiffness_matrix = np.zeros((n_dof, n_dof))
            for i in range(n_dof):
                for j in range(n_dof):
                    if i == j:
                        self.stiffness_matrix[i, i] = (
                            2 * self.params['story_stiffness'] 
                            if i > 0 and i < n_dof-1
                            else self.params['story_stiffness']
                        )
                        if i == 0:
                            self.stiffness_matrix[i, i] += self.params.get('isolator_stiffness', 0)
                    elif abs(i - j) == 1:
                        self.stiffness_matrix[i, j] = -self.params['story_stiffness']
                        self.stiffness_matrix[j, i] = -self.params['story_stiffness']
                        
            # Демпфирование по Рэлею
            alpha = self.params.get('rayleigh_alpha', 0.1)
            beta = self.params.get('rayleigh_beta', 0.1)
            self.damping_matrix = (
                alpha * self.mass_matrix + 
                beta * self.stiffness_matrix
            )
            
        except KeyError as e:
            logger.error(f"Missing parameter in building_params: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error in matrix initialization: {str(e)}")
            raise

    def compute_natural_frequencies(self) -> np.ndarray:
        """
        Вычисление собственных частот системы
        
        Returns:
            np.ndarray: массив собственных частот
        """
        try:
            eigenvalues, _ = eigh(
                a=self.stiffness_matrix, 
                b=self.mass_matrix,
                check_finite=True
            )
            
            self.natural_frequencies = np.sqrt(np.abs(eigenvalues)) / (2 * np.pi)
            return self.natural_frequencies
            
        except np.linalg.LinAlgError as e:
            logger.error(f"Linear algebra error in frequency computation: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error computing natural frequencies: {str(e)}")
            raise