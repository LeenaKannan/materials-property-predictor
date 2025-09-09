
# configuration management for the Materials Property Predictor application.

import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # application settings with environment variable support.
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = False
    api_reload: bool = False
    
    # Database Configuration
    database_url: Optional[str] = None
    
    # Redis Configuration
    redis_url: str = "redis://localhost:6379"
    redis_password: Optional[str] = None
    
    # Materials Project API
    materials_project_api_key: Optional[str] = None
    
    # Model Configuration
    model_cache_dir: str = "./models"
    feature_cache_ttl: int = 3600  # 1 hour in seconds
    
    # Security
    api_key_header: str = "X-API-Key"
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600  # 1 hour in seconds
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()