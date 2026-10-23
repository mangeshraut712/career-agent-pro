"""Configuration management for CareerAgentPro backend."""
import os
from typing import List
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with validation."""
    
    # API Configuration
    api_title: str = "CareerAgentPro API"
    api_version: str = "1.0.0"
    api_description: str = "AI-Powered Career Platform API"
    
    # Security
    secret_key: str = ""
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # CORS Configuration
    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "https://ai-job-helper-steel.vercel.app",
    ]
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    cors_allow_headers: List[str] = ["*"]
    
    # AI Service Configuration
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    ai_model: str = "grok"
    ai_temperature: float = 0.15
    
    # File Upload Configuration
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_file_types: List[str] = [".pdf", ".docx", ".doc", ".txt"]
    
    # Database Configuration
    database_url: str = "sqlite:///./sql_app.db"
    
    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 60
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_format: str = "json"  # json or text
    
    # Environment
    environment: str = os.getenv("ENVIRONMENT", "development")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

