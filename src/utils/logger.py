import logging
from pathlib import Path
from datetime import datetime

def setup_logger(log_path: Path, level: int = logging.INFO):
    """Настройка логирования"""
    log_path = Path(log_path)
    log_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y")