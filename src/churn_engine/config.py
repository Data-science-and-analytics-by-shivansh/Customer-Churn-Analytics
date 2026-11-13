"""Configuration and safe directory-aware logging setup."""

import logging
from datetime import datetime
from pathlib import Path
import yaml


class ChurnConfig:
    """YAML-backed pipeline configuration wrapper."""

    def __init__(self, config_path: str = "config/churn_config.yaml"):
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with path.open("r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f) or {}

    def get(self, key: str, default=None):
        return self.cfg.get(key, default)


def setup_logging(log_level: str = "INFO", log_dir: str = "logs") -> logging.Logger:
    """Configure console + file logging, safely creating log directories first."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("churn_engine")
    if not logger.handlers:
        logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        file_handler = logging.FileHandler(
            Path(log_dir) / f"churn_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    return logger
