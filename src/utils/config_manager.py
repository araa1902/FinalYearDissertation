import yaml
from pathlib import Path
from typing import Dict, Any


class ConfigManager:
    """Managing project configuration"""
    
    _instance = None
    _config = None
    _config_path = "config/config.yaml"
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def load_config(cls, config_path: str = None) -> Dict[str, Any]:
        if cls._config is None:
            path = config_path or cls._config_path
            if not Path(path).exists():
                raise FileNotFoundError(f"Configuration file not found at {path}")
            
            with open(path, "r") as file:
                cls._config = yaml.safe_load(file)
        
        return cls._config
    
    @classmethod
    def reset(cls):
        """Reset configuration"""
        cls._config = None


def load_config(config_path: str = None) -> Dict[str, Any]:
    return ConfigManager.load_config(config_path)
