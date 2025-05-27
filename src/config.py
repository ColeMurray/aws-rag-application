"""
Centralized configuration management for the RAG pipeline.

This module uses Pydantic Settings to manage environment variables and 
application configuration in a type-safe manner.
"""
import os
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # AWS Configuration
    aws_region: str = Field(default="us-east-1", env="AWS_REGION")
    aws_access_key_id: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    
    # Bedrock Model Configuration
    embed_model_id: str = Field(
        default="amazon.titan-embed-text-v2:0",
        env="BEDROCK_EMBED_MODEL_ID",
        description="Bedrock embedding model ID"
    )
    llm_model_id: str = Field(
        default="us.anthropic.claude-sonnet-4-20250514-v1:0",
        env="BEDROCK_LLM_MODEL_ID",
        description="Bedrock LLM model ID"
    )
    
    # Pinecone Configuration
    pinecone_api_key: str = Field(..., env="PINECONE_API_KEY")
    pinecone_environment: str = Field(..., env="PINECONE_ENVIRONMENT")
    pinecone_index_name: str = Field(default="rag-demo-index", env="PINECONE_INDEX_NAME")
    
    # Text Processing Configuration
    chunk_size: int = Field(
        default=800,
        env="CHUNK_SIZE",
        description="Number of characters per text chunk"
    )
    chunk_overlap: int = Field(
        default=100,
        env="CHUNK_OVERLAP",
        description="Overlap between consecutive chunks"
    )
    
    # Retrieval Configuration
    top_k: int = Field(
        default=6,
        env="TOP_K",
        description="Number of chunks to retrieve for each query"
    )
    similarity_threshold: float = Field(
        default=0.5,
        env="SIMILARITY_THRESHOLD",
        description="Minimum similarity score for retrieved chunks"
    )
    
    # LLM Generation Configuration
    max_tokens: int = Field(
        default=512,
        env="MAX_TOKENS",
        description="Maximum tokens for LLM response"
    )
    temperature: float = Field(
        default=0.2,
        env="TEMPERATURE",
        description="Temperature for LLM generation"
    )
    
    # Data Source Configuration
    s3_bucket_name: Optional[str] = Field(default=None, env="S3_BUCKET_NAME")
    data_directory: str = Field(
        default="./data",
        env="DATA_DIRECTORY",
        description="Local directory for document storage"
    )
    
    # API Configuration
    api_title: str = Field(default="RAG Pipeline API", env="API_TITLE")
    api_description: str = Field(
        default="A Retrieval-Augmented Generation API using AWS Bedrock and Pinecone",
        env="API_DESCRIPTION"
    )
    api_version: str = Field(default="1.0.0", env="API_VERSION")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")  # json or console
    
    # Development Configuration
    debug: bool = Field(default=False, env="DEBUG")
    reload: bool = Field(default=False, env="RELOAD")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings.
    
    Uses LRU cache to ensure settings are loaded only once and reused
    throughout the application lifecycle.
    
    Returns:
        Settings: Configured application settings
    """
    return Settings()


# Convenience function for common use cases
def get_aws_region() -> str:
    """Get the configured AWS region."""
    return get_settings().aws_region


def get_pinecone_config() -> dict:
    """Get Pinecone connection configuration."""
    settings = get_settings()
    return {
        "api_key": settings.pinecone_api_key,
        "environment": settings.pinecone_environment,
        "index_name": settings.pinecone_index_name
    }


def get_bedrock_config() -> dict:
    """Get Bedrock model configuration."""
    settings = get_settings()
    return {
        "region": settings.aws_region,
        "embed_model_id": settings.embed_model_id,
        "llm_model_id": settings.llm_model_id
    } 