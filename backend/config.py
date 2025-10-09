"""Configuration management for Materials Property Predictor."""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Materials Project API
    mp_api_key: str = Field(default="", env="MP_API_KEY")
    
    # Redis Configuration
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_password: str = Field(default="", env="REDIS_PASSWORD")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_reload: bool = Field(default=True, env="API_RELOAD")
    
    # Model Configuration
    model_path: str = Field(default="./models", env="MODEL_PATH")
    model_cache_size: int = Field(default=5, env="MODEL_CACHE_SIZE")
    
    # Rate Limiting
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Feature Cache
    feature_cache_ttl: int = Field(default=86400, env="FEATURE_CACHE_TTL")
    
    # Supported Properties
    supported_properties: List[str] = [
        "band_gap",
        "formation_energy",
        "density",
        "e_above_hull"
    ]
    
    # Model Architecture
    ann_hidden_layers: List[int] = [256, 128, 64]
    ann_dropout_rate: float = 0.3
    ann_learning_rate: float = 0.001
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()