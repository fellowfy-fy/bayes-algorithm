# utils/data_storage.py

import json
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

class CustomJSONEncoder(json.JSONEncoder):
    """Пользовательский энкодер для JSON с поддержкой datetime и numpy"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)

class DataStorage:
    """Класс для работы с хранением данных"""
    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
    def save_dataframe(self, df: pd.DataFrame, filename: str):
        """Сохранение DataFrame"""
        try:
            file_path = self.storage_path / filename
            df.to_parquet(file_path)
            logger.info(f"Successfully saved DataFrame to {file_path}")
        except Exception as e:
            logger.error(f"Error saving DataFrame: {str(e)}")
            raise
            
    def load_dataframe(self, filename: str) -> pd.DataFrame:
        """Загрузка DataFrame"""
        try:
            file_path = self.storage_path / filename
            if not file_path.exists():
                return pd.DataFrame()
            return pd.read_parquet(file_path)
        except Exception as e:
            logger.error(f"Error loading DataFrame: {str(e)}")
            raise
            
    def save_json(self, data: dict, filename: str):
        """Сохранение JSON данных"""
        try:
            file_path = self.storage_path / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, cls=CustomJSONEncoder, ensure_ascii=False)
            logger.info(f"Successfully saved JSON to {file_path}")
        except Exception as e:
            logger.error(f"Error saving JSON: {str(e)}")
            raise
            
    def load_json(self, filename: str) -> dict:
        """Загрузка JSON данных"""
        try:
            file_path = self.storage_path / filename
            if not file_path.exists():
                return {}
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON: {str(e)}")
            raise