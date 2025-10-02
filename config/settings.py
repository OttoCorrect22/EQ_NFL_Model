"""
Configuration settings for EQ NFL Model
Centralizes all configurable parameters in one location
"""

import os
from pathlib import Path
from typing import List
from dataclasses import dataclass

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

@dataclass
class DataConfig:
    """Data collection and storage configuration"""
    
    # Seasons to include in model training
    # Default includes last 3 seasons for recency
    SEASONS: List[int] = None
    
    # Data directories
    RAW_DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"
    PROCESSED_DATA_DIR: Path = PROJECT_ROOT / "data" / "processed"
    PREDICTIONS_DIR: Path = PROJECT_ROOT / "data" / "predictions"
    
    # nflreadpy cache settings
    CACHE_MODE: str = "filesystem"  # Options: "memory", "filesystem", "off"
    CACHE_DIR: Path = RAW_DATA_DIR / "cache"
    CACHE_DURATION: int = 86400  # 24 hours in seconds
    
    def __post_init__(self):
        """Initialize default seasons if not specified"""
        if self.SEASONS is None:
            # Dynamically determine current season
            import nflreadpy as nfl
            current_season = nfl.get_current_season()
            # Use last 3 seasons by default
            self.SEASONS = [current_season - 2, current_season - 1, current_season]
        
        # Create directories if they don't exist
        self.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class ModelConfig:
    """Machine learning model configuration"""
    
    # Random seed for reproducibility
    RANDOM_STATE: int = 42
    
    # Train/test split ratios
    TEST_SIZE: float = 0.2
    
    # Model parameters
    N_ESTIMATORS: int = 300  # Number of trees in Random Forest
    MAX_DEPTH: int = 20      # Maximum tree depth
    
    # Feature weighting for recent data
    # Higher values weight recent seasons more heavily
    SEASON_WEIGHTS: dict = None
    
    def __post_init__(self):
        """Initialize default season weights if not specified"""
        if self.SEASON_WEIGHTS is None:
            # Current season: 3x weight, last season: 2x, older: 1x
            self.SEASON_WEIGHTS = {
                'current': 3.0,
                'previous': 2.0,
                'older': 1.0
            }

@dataclass
class PredictionConfig:
    """Prediction output configuration"""
    
    # What to predict
    PREDICT_WINNER: bool = True
    PREDICT_SCORES: bool = True
    PREDICT_MARGIN: bool = True
    PREDICT_TOTAL: bool = True
    
    # Output format
    OUTPUT_FORMAT: str = "csv"  # Options: "csv", "json"
    SAVE_PREDICTIONS: bool = True
    
    # Display settings
    SHOW_CONFIDENCE: bool = True
    SHOW_KEY_FACTORS: bool = True
    MIN_CONFIDENCE_THRESHOLD: float = 0.50  # Only show predictions above this confidence

@dataclass  
class FeatureConfig:
    """Feature engineering configuration"""
    
    # Team statistics to include
    INCLUDE_PASSING_STATS: bool = True
    INCLUDE_RUSHING_STATS: bool = True
    INCLUDE_DEFENSIVE_STATS: bool = True
    INCLUDE_SPECIAL_TEAMS: bool = True
    
    # Advanced metrics
    INCLUDE_EPA: bool = True  # Expected Points Added
    INCLUDE_EFFICIENCY: bool = True
    
    # Recent form calculation
    RECENT_GAMES_LOOKBACK: int = 4  # Number of recent games to analyze
    
    # Context features
    INCLUDE_HOME_ADVANTAGE: bool = True
    INCLUDE_REST_DAYS: bool = True

# Initialize configuration objects
data_config = DataConfig()
model_config = ModelConfig()
prediction_config = PredictionConfig()
feature_config = FeatureConfig()

# Paths helper class
class Paths:
    """Quick access to important paths"""
    ROOT = PROJECT_ROOT
    DATA = data_config.RAW_DATA_DIR.parent
    RAW = data_config.RAW_DATA_DIR
    PROCESSED = data_config.PROCESSED_DATA_DIR
    PREDICTIONS = data_config.PREDICTIONS_DIR
    MODELS = PROJECT_ROOT / "models" / "trained"
    
    @classmethod
    def ensure_directories(cls):
        """Create all necessary directories"""
        for path in [cls.DATA, cls.RAW, cls.PROCESSED, cls.PREDICTIONS, cls.MODELS]:
            path.mkdir(parents=True, exist_ok=True)

# Export commonly used objects
__all__ = [
    'data_config',
    'model_config',
    'prediction_config',
    'feature_config',
    'Paths'
]