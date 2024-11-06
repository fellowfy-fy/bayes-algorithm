import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

class SurrogateModel:
    """Quadratic Response Surface Model as described in paper"""
    def __init__(self, n_params: int):
        self.n_params = n_params
        self.coefficients = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit quadratic response surface according to paper's RSM"""
        X_quad = self._create_quadratic_terms(X)
        # Fit separate models for each frequency
        self.coefficients = []
        for i in range(y.shape[1]):
            coef = np.linalg.lstsq(X_quad, y[:, i], rcond=None)[0]
            self.coefficients.append(coef)
        self.coefficients = np.array(self.coefficients)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict frequencies using calibrated surrogate model"""
        if self.coefficients is None:
            raise ValueError("Model must be fitted before prediction")
        X_quad = self._create_quadratic_terms(X)
        predictions = []
        for coef in self.coefficients:
            pred = X_quad @ coef
            predictions.append(pred)
        return np.array(predictions)
    
    def _create_quadratic_terms(self, X: np.ndarray) -> np.ndarray:
        """Create quadratic terms according to paper's RSM methodology"""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n_samples = X.shape[0]
        
        # Create terms as per paper:
        # [1, h1, h2, h1^2, h2^2, h1*h2]
        quad_terms = np.zeros((n_samples, 6))
        quad_terms[:, 0] = 1  # bias
        quad_terms[:, 1:3] = X  # linear terms
        quad_terms[:, 3] = X[:, 0]**2  # h1^2
        quad_terms[:, 4] = X[:, 1]**2  # h2^2
        quad_terms[:, 5] = X[:, 0] * X[:, 1]  # interaction
        
        return quad_terms

class IsolatorParameters:
    """Class for handling isolator parameters as in paper"""
    def __init__(self):
        # Nominal values from paper
        self.lrb_a_nominal = 7.118E7  # kN/m
        self.lrb_b_nominal = 0.8326E7  # kN/m
        self.sd_hc_nominal = 400.7E7  # kN/m
        self.sd_lc_nominal = 133.1E7  # kN/m
        
    def get_stiffness(self, h1: float, h2: float) -> Dict[str, float]:
        """Calculate isolator stiffness based on parameters"""
        return {
            'lrb_a': self.lrb_a_nominal * h1,
            'lrb_b': self.lrb_b_nominal * h1,
            'sd_hc': self.sd_hc_nominal * h2,
            'sd_lc': self.sd_lc_nominal * h2
        }

class ContinuousBayesianUpdating:
    def __init__(self, n_params: int, prior_mean: np.ndarray, prior_cov: np.ndarray):
        self.n_params = n_params
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.surrogate = SurrogateModel(n_params)
        self.isolator_params = IsolatorParameters()
        self._initialize_surrogate()
        
    def _initialize_surrogate(self):
        """Initialize surrogate model with training data"""
        # Generate training samples around prior mean
        n_samples = 50
        np.random.seed(42)  # For reproducibility
        X_train = np.random.multivariate_normal(self.prior_mean, self.prior_cov, n_samples)
        
        # Generate responses using initial frequencies
        y_train = np.zeros((n_samples, 2))  # Two frequencies
        for i, x in enumerate(X_train):
            # Use quadratic relationship for training data
            f1 = 7.12 * (x[0] + 0.1*x[0]**2 - 0.05*x[0]*x[1])
            f2 = 10.86 * (x[1] + 0.1*x[1]**2 - 0.05*x[0]*x[1])
            y_train[i] = [f1, f2]
            
        self.surrogate.fit(X_train, y_train)
        
    def likelihood(self, theta: np.ndarray, measured_data: np.ndarray, 
                  sigma: float) -> float:
        """Calculate likelihood according to paper's equation"""
        predicted = self.surrogate.predict(theta.reshape(1, -1)).flatten()
        return np.prod([stats.norm.pdf(m - p, 0, sigma) 
                       for m, p in zip(measured_data, predicted)])
    
    def mcmc_metropolis(self, measured_data: np.ndarray, n_iterations: int, 
                       sigma: float) -> Tuple[np.ndarray, np.ndarray]:
        """MCMC using Metropolis algorithm as in paper"""
        current = np.random.multivariate_normal(self.prior_mean, self.prior_cov)
        chain = np.zeros((n_iterations, self.n_params))
        acceptance = np.zeros(n_iterations)
        
        for i in range(n_iterations):
            # Propose new value
            proposal = np.random.multivariate_normal(current, 0.1 * self.prior_cov)
            
            # Calculate likelihoods
            current_likelihood = self.likelihood(current, measured_data, sigma)
            proposal_likelihood = self.likelihood(proposal, measured_data, sigma)
            
            # Calculate prior probabilities
            current_prior = stats.multivariate_normal.pdf(
                current, self.prior_mean, self.prior_cov)
            proposal_prior = stats.multivariate_normal.pdf(
                proposal, self.prior_mean, self.prior_cov)
            
            # Calculate acceptance ratio
            ratio = (proposal_likelihood * proposal_prior) / \
                   (current_likelihood * current_prior)
            
            # Accept/reject
            if np.random.random() < min(1, ratio):
                current = proposal
                acceptance[i] = 1
                
            chain[i] = current
            
        return chain, acceptance

    def plot_results(self, time_intervals: List[str], results: Dict):
        """Plot comprehensive results of the Bayesian updating"""
        # Plot 1: Parameter evolution over time
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        intervals = np.arange(len(time_intervals))
        
        # Plot h1 evolution
        means_h1 = [res['posterior_mean'][0] for res in results.values()]
        stds_h1 = [np.sqrt(res['posterior_cov'][0,0]) for res in results.values()]
        ax1.errorbar(intervals, means_h1, yerr=stds_h1, fmt='o-', capsize=5)
        ax1.set_xlabel('Time Interval')
        ax1.set_ylabel('h1 (LRB stiffness)')
        ax1.set_title('Evolution of LRB Stiffness Parameter')
        ax1.grid(True)
        
        # Plot h2 evolution
        means_h2 = [res['posterior_mean'][1] for res in results.values()]
        stds_h2 = [np.sqrt(res['posterior_cov'][1,1]) for res in results.values()]
        ax2.errorbar(intervals, means_h2, yerr=stds_h2, fmt='o-', capsize=5)
        ax2.set_xlabel('Time Interval')
        ax2.set_ylabel('h2 (SD stiffness)')
        ax2.set_title('Evolution of SD Stiffness Parameter')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Plot 2: Frequency tracking
        plt.figure(figsize=(12, 6))
        f1_means = []
        f2_means = []
        for interval in time_intervals:
            freqs = results[interval]['frequencies']
            f1_means.append(np.mean(freqs[0]))
            f2_means.append(np.mean(freqs[1]))
            
        plt.plot(intervals, f1_means, 'o-', label='f1 (7.12 Hz)')
        plt.plot(intervals, f2_means, 's-', label='f2 (10.86 Hz)')
        plt.xlabel('Time Interval')
        plt.ylabel('Frequencies (Hz)')
        plt.title('Evolution of Natural Frequencies')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Plot 3: MCMC convergence for last interval
        plt.figure(figsize=(12, 6))
        last_chain = results[time_intervals[-1]]['chain']
        plt.plot(last_chain[:, 0], label='h1', alpha=0.7)
        plt.plot(last_chain[:, 1], label='h2', alpha=0.7)
        plt.xlabel('Iteration')
        plt.ylabel('Parameter Value')
        plt.title('MCMC Chain Convergence (Last Interval)')
        plt.legend()
        plt.grid(True)
        plt.show()

def run_continuous_updating(initial_params: Dict, measured_data: Dict, 
                          time_intervals: List[str]) -> Dict:
    """Run continuous model updating as described in paper"""
    updater = ContinuousBayesianUpdating(
        n_params=initial_params['n_params'],
        prior_mean=initial_params['prior_mean'],
        prior_cov=initial_params['prior_cov']
    )
    
    results = {}
    for interval in time_intervals:
        print(f"Processing interval: {interval}")
        
        # Get mean values for current interval
        current_data = np.array([
            np.mean(measured_data[interval][0]),
            np.mean(measured_data[interval][1])
        ])
        
        # Run MCMC
        chain, acceptance = updater.mcmc_metropolis(
            current_data,
            n_iterations=1000,
            sigma=0.1
        )
        
        # Update prior for next interval
        updater.prior_mean = np.mean(chain[500:], axis=0)
        updater.prior_cov = np.cov(chain[500:].T)
        
        # Store results
        results[interval] = {
            'chain': chain,
            'acceptance_rate': acceptance.mean(),
            'posterior_mean': updater.prior_mean,
            'posterior_cov': updater.prior_cov,
            'frequencies': measured_data[interval],
            'isolator_stiffness': updater.isolator_params.get_stiffness(
                updater.prior_mean[0], 
                updater.prior_mean[1]
            )
        }
        
        print(f"Acceptance rate: {acceptance.mean():.2f}")
        print(f"Updated parameters: h1={updater.prior_mean[0]:.3f}, h2={updater.prior_mean[1]:.3f}")
    
    return results

def manual_input_data():
    """Функция для ручного ввода данных измерений"""
    print("\nВвод данных измерений")
    print("----------------------")
    
    time_intervals = []
    measured_data = {}
    
    while True:
        print("\nВыберите действие:")
        print("1. Добавить измерение")
        print("2. Завершить ввод")
        choice = input("Ваш выбор: ")
        
        if choice == '1':
            # Ввод временного интервала
            while True:
                date = input("Введите дату измерений (формат: YYYY-MM): ")
                try:
                    year, month = map(int, date.split('-'))
                    if 2000 <= year <= 2100 and 1 <= month <= 12:
                        break
                    else:
                        print("Некорректная дата. Попробуйте снова.")
                except:
                    print("Неверный формат даты. Используйте YYYY-MM")
            
            # Ввод частот
            try:
                print("\nВведите измеренные частоты:")
                print("Номинальные значения: f1 ≈ 7.12 Hz, f2 ≈ 10.86 Hz")
                f1 = float(input("Введите первую частоту f1 (Hz): "))
                f2 = float(input("Введите вторую частоту f2 (Hz): "))
                
                # Создаем синтетические часовые данные вокруг введенных значений
                n_samples = 720  # 30 дней по часам
                f1_data = f1 + np.random.normal(0, 0.05 * f1, n_samples)
                f2_data = f2 + np.random.normal(0, 0.05 * f2, n_samples)
                
                time_intervals.append(date)
                measured_data[date] = np.array([f1_data, f2_data])
                
                print(f"\nДанные за {date} успешно добавлены")
                print(f"Средние значения: f1 = {np.mean(f1_data):.2f} Hz, f2 = {np.mean(f2_data):.2f} Hz")
                
            except ValueError:
                print("Ошибка ввода. Используйте числовые значения.")
                continue
                
        elif choice == '2':
            if len(time_intervals) < 1:
                print("Необходимо ввести хотя бы одно измерение!")
                continue
            break
        else:
            print("Неверный выбор. Попробуйте снова.")
    
    return time_intervals, measured_data

if __name__ == "__main__":
    # Initialize parameters
    initial_params = {
        'n_params': 2,
        'prior_mean': np.array([1.0, 1.0]),
        'prior_cov': np.array([[0.04, 0], [0, 0.04]]),
        'sampling_freq': 100
    }
    
    print("Байесовский анализ состояния изоляторов")
    print("=======================================")
    print("\nВыберите режим работы:")
    print("1. Ручной ввод данных")
    print("2. Использовать синтетические данные")
    
    mode = input("Ваш выбор: ")
    
    if mode == '1':
        # Ручной ввод данных
        time_intervals, measured_data = manual_input_data()
    else:
        # Использование синтетических данных (оригинальный код)
        time_intervals = [f'2018-{month:02d}' for month in range(9, 13)] + \
                        [f'2019-{month:02d}' for month in range(1, 4)]
        
        measured_data = {}
        np.random.seed(42)
        
        degradation_h1 = np.linspace(1.0, 0.95, len(time_intervals))
        degradation_h2 = np.linspace(1.0, 0.98, len(time_intervals))
        
        for i, interval in enumerate(time_intervals):
            f1 = 7.12 * degradation_h1[i] + np.random.normal(0, 0.356, 720)
            f2 = 10.86 * degradation_h2[i] + np.random.normal(0, 0.543, 720)
            measured_data[interval] = np.array([f1, f2])
    
    # Run analysis
    print("\nЗапуск анализа...")
    results = run_continuous_updating(initial_params, measured_data, time_intervals)
    
    # Plot results
    print("\nПостроение графиков...")
    updater = ContinuousBayesianUpdating(
        initial_params['n_params'],
        initial_params['prior_mean'],
        initial_params['prior_cov']
    )
    updater.plot_results(time_intervals, results)
    
    # Print final results
    print("\nИтоговые результаты:")
    print("===================")
    for interval in time_intervals:
        print(f"\nПериод: {interval}")
        print(f"Параметр h1 (LRB): {results[interval]['posterior_mean'][0]:.3f}")
        print(f"Параметр h2 (SD): {results[interval]['posterior_mean'][1]:.3f}")
        print(f"Средние частоты: f1={np.mean(results[interval]['frequencies'][0]):.2f} Hz, "
              f"f2={np.mean(results[interval]['frequencies'][1]):.2f} Hz")