"""
Configuration settings for the Dynamic Workforce Skill Evolution Analyzer.
"""

import os
import json
from typing import Dict, Any

# Project paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(DATA_DIR, "models")

# Scraping configuration
SCRAPING_CONFIG = {
    "linkedin": {
        "base_url": "https://www.linkedin.com/jobs/search",
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "max_pages": 10,
        "delay": 2,
        "timeout": 10
    },
    "indeed": {
        "base_url": "https://www.indeed.com/jobs",
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "max_pages": 10,
        "delay": 2,
        "timeout": 10
    },
    "glassdoor": {
        "base_url": "https://www.glassdoor.com/Job/jobs.htm",
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "max_pages": 10,
        "delay": 2,
        "timeout": 10
    }
}

# Processing configuration
PROCESSING_CONFIG = {
    "min_token_length": 3,
    "max_token_length": 50,
    "min_skill_occurrences": 5,
    "vectorization": "tfidf",
    "skill_extraction": "hybrid"
}

# Model configuration
MODEL_CONFIG = {
    "skill_extraction": {
        "model_type": "hybrid",
        "bert_model": "bert-base-uncased",
        "threshold": 0.5,
        "batch_size": 16
    },
    "skill_relationships": {
        "min_cooccurrence": 5,
        "min_relationship_strength": 0.1,
        "clustering_algorithm": "louvain"
    },
    "time_series": {
        "seasonality_mode": "multiplicative",
        "forecast_periods": 12,
        "confidence_interval": 0.9,
        "trend_threshold": 0.05
    },
    "skill_demand": {
        "model_type": "random_forest",
        "test_size": 0.2,
        "random_state": 42,
        "n_estimators": 100,
        "max_depth": 10
    },
    "career_path": {
        "relevance_weight": 0.6,
        "trend_weight": 0.4,
        "min_skill_relevance": 0.3
    }
}

# Visualization configuration
VISUALIZATION_CONFIG = {
    "theme": "plotly_white",
    "color_palette": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"],
    "title_font_size": 18,
    "axis_font_size": 14,
    "tick_font_size": 12,
    "legend_font_size": 12
}

# Web application configuration
APP_CONFIG = {
    "host": "0.0.0.0",
    "port": 8050,
    "debug": True,
    "title": "Dynamic Workforce Skill Evolution Analyzer",
    "description": "A comprehensive career intelligence platform that tracks skill demands across industries and predicts future trends."
}

# Default data
DEFAULT_DATA = {
    "queries": ["data scientist", "data analyst", "machine learning engineer", "software engineer", "product manager"],
    "locations": ["New York", "San Francisco", "Seattle", "Austin", "Remote"],
    "sources": ["linkedin", "indeed"]
}

# Load custom configuration if available
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r") as f:
            custom_config = json.load(f)
            
            # Update configurations with custom values
            if "scraping" in custom_config:
                for source, config in custom_config["scraping"].items():
                    if source in SCRAPING_CONFIG:
                        SCRAPING_CONFIG[source].update(config)
            
            if "processing" in custom_config:
                PROCESSING_CONFIG.update(custom_config["processing"])
            
            if "model" in custom_config:
                for model, config in custom_config["model"].items():
                    if model in MODEL_CONFIG:
                        MODEL_CONFIG[model].update(config)
            
            if "visualization" in custom_config:
                VISUALIZATION_CONFIG.update(custom_config["visualization"])
            
            if "app" in custom_config:
                APP_CONFIG.update(custom_config["app"])
            
            if "data" in custom_config:
                DEFAULT_DATA.update(custom_config["data"])
    
    except Exception as e:
        print(f"Error loading custom configuration: {str(e)}")