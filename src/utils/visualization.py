import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class MonitoringVisualizer:
    """Класс для визуализации результатов мониторинга"""
    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
    def plot_time_series(self, data: pd.DataFrame, parameter: str,
                        title: Optional[str] = None):
        """Построение временного ряда"""
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=data, x='timestamp', y=parameter)
        
        if title:
            plt.title(title)
        plt.xlabel('Time')
        plt.ylabel(parameter)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        file_path = self.output_path / f"{parameter}_time_series.png"
        plt.savefig(file_path)
        plt.close()
        
    def plot_correlation_matrix(self, data: pd.DataFrame):
        """Построение матрицы корреляций"""
        plt.figure(figsize=(10, 8))
        correlation_matrix = data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        
        file_path = self.output_path / "correlation_matrix.png"
        plt.savefig(file_path)
        plt.close()
        
    def plot_damage_indicators(self, data: pd.DataFrame):
        """Визуализация индикаторов повреждений"""
        damage_columns = [col for col in data.columns if 'damage' in col]
        if not damage_columns:
            return
            
        plt.figure(figsize=(12, 6))
        for col in damage_columns:
            sns.lineplot(data=data, x='timestamp', y=col, label=col)
            
        plt.title('Damage Indicators Over Time')
        plt.xlabel('Time')
        plt.ylabel('Indicator Value')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        
        file_path = self.output_path / "damage_indicators.png"
        plt.savefig(file_path)
        plt.close()