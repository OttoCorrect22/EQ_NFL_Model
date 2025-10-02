"""
Team Statistics Aggregator
Converts game-by-game statistics into weighted team averages
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import logging

from config.settings import model_config

logger = logging.getLogger(__name__)


class TeamAggregator:
    """
    Aggregates team statistics with recency weighting
    
    Takes game-by-game team performance data and creates team profiles
    by calculating weighted averages. Recent games are weighted more heavily
    to capture current team strength.
    
    Weighting (from config):
    - Current season: 3x
    - Previous season: 2x  
    - Older seasons: 1x
    """
    
    def __init__(self):
        """Initialize aggregator with configuration"""
        self.season_weights = model_config.SEASON_WEIGHTS
        self.current_season = None  # Will be set when aggregating
        
    def _determine_weight(self, season: int) -> float:
        """
        Determine weight for a given season
        
        Args:
            season: Year of the season
            
        Returns:
            Weight multiplier for that season
        """
        if self.current_season is None:
            raise ValueError("Current season not set")
        
        if season == self.current_season:
            return self.season_weights['current']
        elif season == self.current_season - 1:
            return self.season_weights['previous']
        else:
            return self.season_weights['older']
    
    def aggregate_team_stats(self, team_stats: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate team statistics with season weighting
        
        Args:
            team_stats: DataFrame with game-by-game team statistics
                       Must have columns: team, season, week
                       
        Returns:
            DataFrame with one row per team containing weighted averages
            
        Example:
            aggregator = TeamAggregator()
            team_averages = aggregator.aggregate_team_stats(team_stats)
            buf_avg = team_averages.loc['BUF']
        """
        logger.info("Aggregating team statistics with season weighting")
        
        # Determine current season (most recent in data)
        self.current_season = team_stats['season'].max()
        logger.info(f"Current season identified as: {self.current_season}")
        
        # Define statistics to aggregate
        stats_to_aggregate = self._get_stats_list(team_stats)
        logger.info(f"Aggregating {len(stats_to_aggregate)} statistics")
        
        # Calculate weights for each game
        team_stats = team_stats.copy()
        team_stats['weight'] = team_stats['season'].apply(self._determine_weight)
        
        # Group by team and calculate weighted averages
        team_averages_list = []
        
        for team in team_stats['team'].unique():
            team_data = team_stats[team_stats['team'] == team].copy()
            
            team_avg = {'team': team}
            
            # Calculate weighted average for each stat
            for stat in stats_to_aggregate:
                if stat in team_data.columns:
                    # Handle potential NaN values
                    valid_data = team_data[[stat, 'weight']].dropna()
                    
                    if len(valid_data) > 0:
                        weighted_avg = np.average(
                            valid_data[stat],
                            weights=valid_data['weight']
                        )
                        team_avg[stat] = round(weighted_avg, 2)
                    else:
                        team_avg[stat] = 0.0
            
            team_averages_list.append(team_avg)
        
        team_averages = pd.DataFrame(team_averages_list).set_index('team')
        
        # Add derived metrics
        team_averages = self._add_derived_metrics(team_averages)
        
        logger.info(f"Aggregated stats for {len(team_averages)} teams")
        logger.info(f"Total features per team: {len(team_averages.columns)}")
        
        return team_averages
    
    def _get_stats_list(self, team_stats: pd.DataFrame) -> List[str]:
        """
        Determine which statistics to aggregate based on available columns
        
        Args:
            team_stats: DataFrame with team statistics
            
        Returns:
            List of column names to aggregate
        """
        # Core offensive stats
        offensive_stats = [
            'passing_yards', 'passing_tds', 'passing_interceptions',
            'passing_epa', 'passing_first_downs',
            'rushing_yards', 'rushing_tds', 'rushing_epa',
            'rushing_first_downs', 'rushing_fumbles_lost',
            'sacks_suffered'
        ]
        
        # Core defensive stats
        defensive_stats = [
            'def_sacks', 'def_interceptions', 'def_tds',
            'def_fumbles_forced', 'def_tackles_for_loss',
            'def_qb_hits'
        ]
        
        # Special teams stats
        special_teams_stats = [
            'fg_made', 'fg_att', 'fg_pct', 'pat_pct'
        ]
        
        # Combine and filter to available columns
        all_stats = offensive_stats + defensive_stats + special_teams_stats
        available_stats = [stat for stat in all_stats if stat in team_stats.columns]
        
        return available_stats
    
    def _add_derived_metrics(self, team_averages: pd.DataFrame) -> pd.DataFrame:
        """
        Add calculated efficiency metrics
        
        Creates composite metrics that combine multiple base statistics
        
        Args:
            team_averages: DataFrame with base statistics
            
        Returns:
            DataFrame with added derived metrics
        """
        df = team_averages.copy()
        
        # Total offense (yards)
        if 'passing_yards' in df.columns and 'rushing_yards' in df.columns:
            df['total_offense'] = df['passing_yards'] + df['rushing_yards']
        
        # Total touchdowns
        if 'passing_tds' in df.columns and 'rushing_tds' in df.columns:
            df['total_tds'] = df['passing_tds'] + df['rushing_tds']
        
        # Total EPA (Expected Points Added)
        if 'passing_epa' in df.columns and 'rushing_epa' in df.columns:
            df['total_epa'] = df['passing_epa'] + df['rushing_epa']
        
        # Offensive efficiency (normalized)
        if 'total_offense' in df.columns:
            df['offensive_efficiency'] = (df['total_offense'] / 350).round(2)
        
        # Defensive efficiency (higher is better)
        if 'def_sacks' in df.columns and 'def_interceptions' in df.columns:
            df['defensive_efficiency'] = (
                (df['def_sacks'] + df['def_interceptions'] * 2) / 10
            ).round(2)
        
        # Turnover differential components
        if 'passing_interceptions' in df.columns and 'rushing_fumbles_lost' in df.columns:
            df['turnovers_lost'] = df['passing_interceptions'] + df['rushing_fumbles_lost']
        
        if 'def_interceptions' in df.columns and 'def_fumbles_forced' in df.columns:
            df['turnovers_gained'] = df['def_interceptions'] + df['def_fumbles_forced']
        
        if 'turnovers_lost' in df.columns and 'turnovers_gained' in df.columns:
            df['turnover_differential'] = df['turnovers_gained'] - df['turnovers_lost']
        
        logger.info(f"Added {len(df.columns) - len(team_averages.columns)} derived metrics")
        
        return df
    
    def get_team_profile(self, team_averages: pd.DataFrame, team: str) -> pd.Series:
        """
        Get full statistical profile for a specific team
        
        Args:
            team_averages: DataFrame from aggregate_team_stats()
            team: Team abbreviation (e.g., 'BUF', 'KC')
            
        Returns:
            Series with all statistics for that team
            
        Example:
            buf_profile = aggregator.get_team_profile(team_averages, 'BUF')
            print(f"BUF total offense: {buf_profile['total_offense']}")
        """
        if team not in team_averages.index:
            raise ValueError(f"Team '{team}' not found in averages")
        
        return team_averages.loc[team]
    
    def compare_teams(self, 
                     team_averages: pd.DataFrame,
                     team1: str,
                     team2: str) -> pd.DataFrame:
        """
        Compare statistics between two teams
        
        Args:
            team_averages: DataFrame from aggregate_team_stats()
            team1: First team abbreviation
            team2: Second team abbreviation
            
        Returns:
            DataFrame showing both teams' stats side by side
            
        Example:
            comparison = aggregator.compare_teams(team_averages, 'BUF', 'KC')
            print(comparison[['total_offense', 'total_epa']])
        """
        if team1 not in team_averages.index:
            raise ValueError(f"Team '{team1}' not found")
        if team2 not in team_averages.index:
            raise ValueError(f"Team '{team2}' not found")
        
        comparison = pd.DataFrame({
            team1: team_averages.loc[team1],
            team2: team_averages.loc[team2]
        })
        
        # Add difference column
        comparison['difference'] = comparison[team1] - comparison[team2]
        
        return comparison


# Convenience function
def aggregate_teams(team_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Quick function to aggregate team statistics
    
    Args:
        team_stats: Game-by-game team statistics
        
    Returns:
        Aggregated team averages
        
    Example:
        from src.features.aggregator import aggregate_teams
        team_averages = aggregate_teams(team_stats)
    """
    aggregator = TeamAggregator()
    return aggregator.aggregate_team_stats(team_stats)