"""
NFL Data Loader
Handles data collection from nflreadpy with caching and configuration
"""

import nflreadpy as nfl
import pandas as pd
from typing import List, Optional, Dict
import logging
from pathlib import Path

from config.settings import data_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NFLDataLoader:
    """
    Loads NFL data from nflreadpy with proper configuration and caching
    
    This class is the single point of contact for all NFL data needs.
    It handles:
    - Season management
    - Data caching
    - Converting between Polars and Pandas
    - Error handling
    
    Usage:
        loader = NFLDataLoader()
        schedules = loader.load_schedules()
        team_stats = loader.load_team_stats()
    """
    
    def __init__(self, seasons: Optional[List[int]] = None):
        """
        Initialize the data loader
        
        Args:
            seasons: List of NFL seasons to load (e.g., [2023, 2024, 2025])
                    If None, uses seasons from config
        """
        # Use provided seasons or fall back to config
        self.seasons = seasons if seasons is not None else data_config.SEASONS
        
        # Configure nflreadpy caching
        # Note: Pass Path object directly, don't convert to string
        from nflreadpy.config import update_config
        update_config(
            cache_mode=data_config.CACHE_MODE,
            cache_dir=data_config.CACHE_DIR,  # Pass Path object directly
            cache_duration=data_config.CACHE_DURATION,
            verbose=False  # Reduce noise in logs
        )
        
        logger.info(f"NFLDataLoader initialized for seasons: {self.seasons}")
    
    def _convert_to_pandas(self, data) -> pd.DataFrame:
        """
        Convert data to pandas DataFrame if needed
        
        nflreadpy returns Polars DataFrames by default, but we use pandas
        for compatibility with scikit-learn
        
        Args:
            data: Data from nflreadpy (might be Polars or Pandas)
            
        Returns:
            Pandas DataFrame
        """
        if hasattr(data, 'to_pandas'):
            return data.to_pandas()
        return data
    
    def load_schedules(self, seasons: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Load game schedules and results
        
        Contains:
        - Game dates and times
        - Teams playing
        - Final scores (for completed games)
        - Betting lines (spreads, totals, moneylines)
        - Game context (weather, referee, stadium)
        
        Args:
            seasons: Seasons to load (uses instance default if None)
            
        Returns:
            DataFrame with game schedules
            
        Example:
            schedules = loader.load_schedules()
            upcoming = schedules[schedules['away_score'].isna()]
        """
        seasons = seasons or self.seasons
        logger.info(f"Loading schedules for seasons: {seasons}")
        
        try:
            schedules = nfl.load_schedules(seasons=seasons)
            schedules = self._convert_to_pandas(schedules)
            
            logger.info(f"Loaded {len(schedules)} games")
            return schedules
            
        except Exception as e:
            logger.error(f"Error loading schedules: {e}")
            raise
    
    def load_team_stats(self, seasons: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Load team-level statistics (game-by-game)
        
        Contains per-game stats for each team:
        - Passing: yards, TDs, interceptions, EPA
        - Rushing: yards, TDs, fumbles, EPA  
        - Defense: sacks, tackles, interceptions
        - Special teams: field goals, punts
        
        Note: Each row is one team's performance in one game
        
        Args:
            seasons: Seasons to load (uses instance default if None)
            
        Returns:
            DataFrame with team statistics
            
        Example:
            team_stats = loader.load_team_stats()
            buf_stats = team_stats[team_stats['team'] == 'BUF']
        """
        seasons = seasons or self.seasons
        logger.info(f"Loading team stats for seasons: {seasons}")
        
        try:
            team_stats = nfl.load_team_stats(seasons=seasons)
            team_stats = self._convert_to_pandas(team_stats)
            
            logger.info(f"Loaded {len(team_stats)} team-game records")
            return team_stats
            
        except Exception as e:
            logger.error(f"Error loading team stats: {e}")
            raise
    
    def load_player_stats(self, seasons: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Load player-level statistics
        
        Contains individual player performance data.
        Currently not used in base model but available for future enhancements.
        
        Args:
            seasons: Seasons to load (uses instance default if None)
            
        Returns:
            DataFrame with player statistics
        """
        seasons = seasons or self.seasons
        logger.info(f"Loading player stats for seasons: {seasons}")
        
        try:
            player_stats = nfl.load_player_stats(seasons=seasons)
            player_stats = self._convert_to_pandas(player_stats)
            
            logger.info(f"Loaded {len(player_stats)} player-game records")
            return player_stats
            
        except Exception as e:
            logger.error(f"Error loading player stats: {e}")
            raise
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all relevant datasets at once
        
        Convenience method for loading all data needed for model training
        
        Returns:
            Dictionary with keys: 'schedules', 'team_stats'
            
        Example:
            data = loader.load_all_data()
            schedules = data['schedules']
            team_stats = data['team_stats']
        """
        logger.info("Loading all data for model training")
        
        data = {
            'schedules': self.load_schedules(),
            'team_stats': self.load_team_stats()
        }
        
        logger.info("All data loaded successfully")
        return data
    
    def get_current_week(self) -> int:
        """
        Get the current NFL week
        
        Returns:
            Current week number (1-18 for regular season)
        """
        try:
            return nfl.get_current_week()
        except Exception as e:
            logger.warning(f"Could not determine current week: {e}")
            return 1
    
    def get_current_season(self) -> int:
        """
        Get the current NFL season year
        
        Returns:
            Current season year (e.g., 2025)
        """
        try:
            return nfl.get_current_season()
        except Exception as e:
            logger.warning(f"Could not determine current season: {e}")
            return 2025
    
    def save_data(self, data: pd.DataFrame, filename: str, directory: str = "raw") -> Path:
        """
        Save DataFrame to parquet file
        
        Parquet is compressed and faster to load than CSV
        
        Args:
            data: DataFrame to save
            filename: Name for the file (without extension)
            directory: Subdirectory in data/ ("raw", "processed", or "predictions")
            
        Returns:
            Path to saved file
            
        Example:
            path = loader.save_data(schedules, "schedules_2025", "raw")
        """
        if directory == "raw":
            save_dir = data_config.RAW_DATA_DIR
        elif directory == "processed":
            save_dir = data_config.PROCESSED_DATA_DIR
        elif directory == "predictions":
            save_dir = data_config.PREDICTIONS_DIR
        else:
            raise ValueError(f"Unknown directory: {directory}")
        
        save_dir.mkdir(parents=True, exist_ok=True)
        filepath = save_dir / f"{filename}.parquet"
        
        data.to_parquet(filepath, index=False)
        logger.info(f"Saved {len(data)} records to {filepath}")
        
        return filepath


# Convenience function for quick access
def get_loader(seasons: Optional[List[int]] = None) -> NFLDataLoader:
    """
    Get a configured data loader instance
    
    Args:
        seasons: Optional season list, uses config default if None
        
    Returns:
        Configured NFLDataLoader instance
        
    Example:
        from src.data.loader import get_loader
        loader = get_loader()
        schedules = loader.load_schedules()
    """
    return NFLDataLoader(seasons=seasons)