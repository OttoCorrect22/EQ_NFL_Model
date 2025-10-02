"""
Recent Form Calculator
Analyzes team performance trends over recent games
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

from config.settings import feature_config

logger = logging.getLogger(__name__)


class RecentFormCalculator:
    """
    Calculates recent team form metrics
    
    Analyzes the last N games (default 4) for each team to capture:
    - Win/loss record
    - Scoring trends
    - Point differentials
    - Performance momentum
    
    Recent form can indicate teams getting hot or cold, which may not
    be captured by season-long averages.
    """
    
    def __init__(self, lookback_games: int = None):
        """
        Initialize calculator
        
        Args:
            lookback_games: Number of recent games to analyze
                          Defaults to config setting (4 games)
        """
        self.lookback_games = (
            lookback_games if lookback_games is not None 
            else feature_config.RECENT_GAMES_LOOKBACK
        )
        logger.info(f"RecentFormCalculator initialized with {self.lookback_games} game lookback")
    
    def calculate_recent_form(self, schedules: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate recent form for all teams
        
        Args:
            schedules: DataFrame with game schedules and results
                      Must have: away_team, home_team, away_score, home_score
                      
        Returns:
            DataFrame with one row per team containing recent form metrics
            
        Example:
            calculator = RecentFormCalculator()
            recent_form = calculator.calculate_recent_form(schedules)
            buf_form = recent_form.loc['BUF']
        """
        logger.info(f"Calculating recent form (last {self.lookback_games} games)")
        
        # Get all teams
        teams = sorted(set(schedules['away_team'].unique()) | set(schedules['home_team'].unique()))
        
        recent_form_data = []
        
        for team in teams:
            form_metrics = self._calculate_team_form(schedules, team)
            recent_form_data.append(form_metrics)
        
        recent_form = pd.DataFrame(recent_form_data).set_index('team')
        
        logger.info(f"Calculated recent form for {len(recent_form)} teams")
        logger.info(f"Form metrics per team: {len(recent_form.columns)}")
        
        return recent_form
    
    def _calculate_team_form(self, schedules: pd.DataFrame, team: str) -> Dict:
        """
        Calculate recent form metrics for a single team
        
        Args:
            schedules: Game schedules DataFrame
            team: Team abbreviation
            
        Returns:
            Dictionary with form metrics
        """
        # Get team's games (both home and away)
        team_games = self._get_team_games(schedules, team)
        
        # Sort by recency and take last N games
        recent_games = team_games.tail(self.lookback_games)
        
        if len(recent_games) == 0:
            return self._empty_form_metrics(team)
        
        # Calculate metrics
        metrics = {
            'team': team,
            'games_played': len(recent_games),
            'wins': recent_games['won'].sum(),
            'losses': len(recent_games) - recent_games['won'].sum(),
            'win_pct': (recent_games['won'].sum() / len(recent_games)).round(3),
            'avg_points_scored': recent_games['points_scored'].mean().round(1),
            'avg_points_allowed': recent_games['points_allowed'].mean().round(1),
            'avg_margin': recent_games['margin'].mean().round(1),
            'total_margin': recent_games['margin'].sum().round(1)
        }
        
        # Add trend indicators
        if len(recent_games) >= 2:
            metrics['scoring_trend'] = self._calculate_trend(
                recent_games['points_scored'].values
            )
            metrics['margin_trend'] = self._calculate_trend(
                recent_games['margin'].values
            )
        else:
            metrics['scoring_trend'] = 0.0
            metrics['margin_trend'] = 0.0
        
        # Recent win streak
        metrics['current_streak'] = self._calculate_streak(recent_games['won'].values)
        
        return metrics
    
    def _get_team_games(self, schedules: pd.DataFrame, team: str) -> pd.DataFrame:
        """
        Get all completed games for a team with outcome information
        
        Args:
            schedules: Game schedules
            team: Team to get games for
            
        Returns:
            DataFrame with team's games including win/loss and scores
        """
        # Filter to completed games only
        completed = schedules[schedules['away_score'].notna()].copy()
        
        games = []
        
        # Away games
        away_games = completed[completed['away_team'] == team].copy()
        for _, game in away_games.iterrows():
            games.append({
                'season': game['season'],
                'week': game['week'],
                'opponent': game['home_team'],
                'location': 'away',
                'points_scored': game['away_score'],
                'points_allowed': game['home_score'],
                'won': 1 if game['away_score'] > game['home_score'] else 0,
                'margin': game['away_score'] - game['home_score']
            })
        
        # Home games
        home_games = completed[completed['home_team'] == team].copy()
        for _, game in home_games.iterrows():
            games.append({
                'season': game['season'],
                'week': game['week'],
                'opponent': game['away_team'],
                'location': 'home',
                'points_scored': game['home_score'],
                'points_allowed': game['away_score'],
                'won': 1 if game['home_score'] > game['away_score'] else 0,
                'margin': game['home_score'] - game['away_score']
            })
        
        if not games:
            return pd.DataFrame()
        
        team_games = pd.DataFrame(games)
        # Sort by season and week
        team_games = team_games.sort_values(['season', 'week'])
        
        return team_games
    
    def _calculate_trend(self, values: np.ndarray) -> float:
        """
        Calculate trend in values (positive = improving, negative = declining)
        
        Uses simple linear regression slope
        
        Args:
            values: Array of values over time
            
        Returns:
            Trend coefficient (slope)
        """
        if len(values) < 2:
            return 0.0
        
        # Simple linear regression
        x = np.arange(len(values))
        
        # Calculate slope
        x_mean = x.mean()
        y_mean = values.mean()
        
        numerator = ((x - x_mean) * (values - y_mean)).sum()
        denominator = ((x - x_mean) ** 2).sum()
        
        if denominator == 0:
            return 0.0
        
        slope = numerator / denominator
        return round(slope, 2)
    
    def _calculate_streak(self, results: np.ndarray) -> int:
        """
        Calculate current win/loss streak
        
        Args:
            results: Array of 1s (wins) and 0s (losses), most recent last
            
        Returns:
            Streak length (positive for wins, negative for losses)
        """
        if len(results) == 0:
            return 0
        
        # Start from most recent game
        current_result = results[-1]
        streak = 1 if current_result == 1 else -1
        
        # Count backwards while result is the same
        for i in range(len(results) - 2, -1, -1):
            if results[i] == current_result:
                streak += 1 if current_result == 1 else -1
            else:
                break
        
        return streak
    
    def _empty_form_metrics(self, team: str) -> Dict:
        """
        Return empty form metrics for teams with no recent games
        
        Args:
            team: Team abbreviation
            
        Returns:
            Dictionary with zero/default values
        """
        return {
            'team': team,
            'games_played': 0,
            'wins': 0,
            'losses': 0,
            'win_pct': 0.0,
            'avg_points_scored': 0.0,
            'avg_points_allowed': 0.0,
            'avg_margin': 0.0,
            'total_margin': 0.0,
            'scoring_trend': 0.0,
            'margin_trend': 0.0,
            'current_streak': 0
        }
    
    def get_team_form(self, recent_form: pd.DataFrame, team: str) -> pd.Series:
        """
        Get form metrics for a specific team
        
        Args:
            recent_form: DataFrame from calculate_recent_form()
            team: Team abbreviation
            
        Returns:
            Series with form metrics
            
        Example:
            buf_form = calculator.get_team_form(recent_form, 'BUF')
            print(f"Buffalo recent wins: {buf_form['wins']}")
        """
        if team not in recent_form.index:
            raise ValueError(f"Team '{team}' not found in recent form data")
        
        return recent_form.loc[team]
    
    def compare_form(self,
                    recent_form: pd.DataFrame,
                    team1: str,
                    team2: str) -> pd.DataFrame:
        """
        Compare recent form between two teams
        
        Args:
            recent_form: DataFrame from calculate_recent_form()
            team1: First team
            team2: Second team
            
        Returns:
            DataFrame comparing both teams' recent form
            
        Example:
            comparison = calculator.compare_form(recent_form, 'BUF', 'KC')
            print(comparison[['wins', 'avg_margin']])
        """
        if team1 not in recent_form.index:
            raise ValueError(f"Team '{team1}' not found")
        if team2 not in recent_form.index:
            raise ValueError(f"Team '{team2}' not found")
        
        comparison = pd.DataFrame({
            team1: recent_form.loc[team1],
            team2: recent_form.loc[team2]
        })
        
        comparison['difference'] = comparison[team1] - comparison[team2]
        
        return comparison


# Convenience function
def calculate_form(schedules: pd.DataFrame, lookback_games: int = None) -> pd.DataFrame:
    """
    Quick function to calculate recent form
    
    Args:
        schedules: Game schedules with results
        lookback_games: Number of recent games (defaults to config)
        
    Returns:
        Recent form DataFrame
        
    Example:
        from src.features.recent_form import calculate_form
        recent_form = calculate_form(schedules)
    """
    calculator = RecentFormCalculator(lookback_games=lookback_games)
    return calculator.calculate_recent_form(schedules)