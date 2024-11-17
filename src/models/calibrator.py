from typing import Dict, List, Tuple
import numpy as np
from scipy.optimize import minimize
import logging
from .fe_model import FEModel

logger = logging.getLogger(__name__)

class ModelCalibrator:
    """Класс для калибровки FE модели"""
    def __init__(self, fe_model: FEModel, measured_frequencies: Tuple[float, float]):
        self.fe_model = fe_model
        self.measured_frequencies = np.array(measured_frequencies)
        self.calibrated_params = None
        
    def objective_function(self, params: np.ndarray) -> float:
        """Целевая функция для оптимизации"""
        try:
            lrb_stiffness, sd_friction = params
            self.fe_model.update_isolator_parameters(lrb_stiffness, sd_friction)
            computed_frequencies = self.fe_model.compute_natural_frequencies()
            rel_error = np.abs(
                (computed_frequencies[:2] - self.measured_frequencies) / 
                self.measured_frequencies
            )
            return np.sum(rel_error**2)
        except Exception as e:
            logger.error(f"Error in objective function: {str(e)}")
            return float('inf')
    
    def calibrate(self, initial_guess: np.ndarray, 
                 bounds: List[Tuple[float, float]]) -> Dict:
        """Калибровка параметров модели"""
        try:
            result = minimize(
                fun=self.objective_function,
                x0=initial_guess,
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-6}
            )
            
            if not result.success:
                logger.warning(f"Optimization did not converge: {result.message}")
            
            self.calibrated_params = {
                'lrb_stiffness': result.x[0],
                'sd_friction': result.x[1],
                'optimization_success': result.success,
                'final_error': result.fun,
                'message': result.message,
                'n_iterations': result.nit
            }
            
            return self.calibrated_params
        except Exception as e:
            logger.error(f"Error in model calibration: {str(e)}")
            raise