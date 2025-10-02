"""
Feature Engineer
Creates game-level features from team statistics and recent form
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import logging

from config.settings import feature_config

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Engineers features for game predictions
    
    Takes team-level statistics and creates game-level features by:
    1. Adding home and away team stats
    2. Calculating differentials (home - away)
    3. Adding context features (home advantage, rest, etc.)
    
    The model learns patterns like "teams with +50 yard differential win 65% of the time"
    """
    
    def __init__(self):
        """Initialize feature engineer with configuration"""
        self.config = feature_config
        
    def create_game_features(self,
                           schedules: pd.DataFrame,
                           team_averages: pd.DataFrame,
                           recent_form: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for all games in schedule
        
        Args:
            schedules: Game schedules (can include completed or upcoming games)
            team_averages: Team statistical averages from aggregator
            recent_form: Recent form metrics from calculator
            
        Returns:
            DataFrame with engineered features for each game
            One row per game with all predictive features
            
        Example:
            engineer = FeatureEngineer()
            features = engineer.create_game_features(
                schedules, team_averages, recent_form
            )
        """
        logger.info(f"Engineering features for {len(schedules)} games")
        
        game_features = []
        skipped = 0
        
        for _, game in schedules.iterrows():
            away_team = game['away_team']
            home_team = game['home_team']
            
            # Skip if missing data for either team
            if not self._has_required_data(away_team, home_team, team_averages, recent_form):
                skipped += 1
                continue
            
            # Create feature dictionary for this game
            features = self._create_single_game_features(
                game, away_team, home_team, team_averages, recent_form
            )
            
            game_features.append(features)
        
        features_df = pd.DataFrame(game_features)
        
        logger.info(f"Created {len(features_df)} game feature vectors (skipped {skipped})")
        logger.info(f"Features per game: {len(features_df.columns)}")
        
        return features_df
    
    def _has_required_data(self,
                          away_team: str,
                          home_team: str,
                          team_averages: pd.DataFrame,
                          recent_form: pd.DataFrame) -> bool:
        """Check if both teams have required data"""
        return (
            away_team in team_averages.index and
            home_team in team_averages.index and
            away_team in recent_form.index and
            home_team in recent_form.index
        )
    
    def _create_single_game_features(self,
                                    game: pd.Series,
                                    away_team: str,
                                    home_team: str,
                                    team_averages: pd.DataFrame,
                                    recent_form: pd.DataFrame) -> Dict:
        """
        Create all features for a single game
        
        Args:
            game: Single game from schedule
            away_team: Away team abbreviation
            home_team: Home team abbreviation
            team_averages: Team statistical averages
            recent_form: Recent form data
            
        Returns:
            Dictionary with all features for this game
        """
        features = {}
        
        # Game identifiers
        features['game_id'] = game.get('game_id', f"{game['season']}_{game['week']}_{away_team}_{home_team}")
        features['season'] = game['season']
        features['week'] = game['week']
        features['away_team'] = away_team
        features['home_team'] = home_team
        
        # Get team data
        away_avg = team_averages.loc[away_team]
        home_avg = team_averages.loc[home_team]
        away_form = recent_form.loc[away_team]
        home_form = recent_form.loc[home_team]
        
        # Add team average features (prefixed with away_ and home_)
        for col in team_averages.columns:
            features[f'away_{col}'] = away_avg[col]
            features[f'home_{col}'] = home_avg[col]
            
            # Calculate differential (home - away)
            # Positive differential favors home team
            features[f'diff_{col}'] = home_avg[col] - away_avg[col]
        
        # Add recent form features
        form_features = ['wins', 'win_pct', 'avg_points_scored', 'avg_points_allowed',
                        'avg_margin', 'scoring_trend', 'margin_trend', 'current_streak']
        
        for feat in form_features:
            if feat in recent_form.columns:
                features[f'away_recent_{feat}'] = away_form[feat]
                features[f'home_recent_{feat}'] = home_form[feat]
                features[f'diff_recent_{feat}'] = home_form[feat] - away_form[feat]
        
        # Context features
        if self.config.INCLUDE_HOME_ADVANTAGE:
            features['home_advantage'] = 1  # Binary indicator
        
        if self.config.INCLUDE_REST_DAYS:
            features['rest_advantage'] = self._calculate_rest_advantage(game)
        
        # Add betting lines if available
        if 'spread_line' in game.index and pd.notna(game['spread_line']):
            features['spread_line'] = game['spread_line']
        else:
            features['spread_line'] = 0.0
        
        if 'total_line' in game.index and pd.notna(game['total_line']):
            features['total_line'] = game['total_line']
        else:
            features['total_line'] = 0.0
        
        # Target variables (if game is completed)
        if 'away_score' in game.index and pd.notna(game['away_score']):
            home_score = float(game['home_score'])
            away_score = float(game['away_score'])
            
            features['home_score'] = home_score
            features['away_score'] = away_score
            features['home_wins'] = 1 if home_score > away_score else 0
            features['margin'] = home_score - away_score
            features['total_points'] = home_score + away_score
        
        return features
    
    def _calculate_rest_advantage(self, game: pd.Series) -> float:
        """
        Calculate rest day advantage (home rest - away rest)
        
        Args:
            game: Game row from schedule
            
        Returns:
            Rest day differential (positive favors home team)
        """
        if 'away_rest' in game.index and 'home_rest' in game.index:
            if pd.notna(game['away_rest']) and pd.notna(game['home_rest']):
                return float(game['home_rest'] - game['away_rest'])
        
        return 0.0
    
    def get_feature_importance_names(self, features_df: pd.DataFrame) -> List[str]:
        """
        Get list of feature names for model training
        
        Excludes identifier and target columns
        
        Args:
            features_df: DataFrame with all features
            
        Returns:
            List of feature column names to use for training
        """
        exclude_cols = [
            'game_id', 'season', 'week', 'away_team', 'home_team',
            'home_score', 'away_score', 'home_wins', 'margin', 'total_points'
        ]
        
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        return feature_cols
    
    def split_features_targets(self, features_df: pd.DataFrame) -> tuple:
        """
        Split features DataFrame into X (features) and y (targets)
        
        Args:
            features_df: Complete features DataFrame
            
        Returns:
            Tuple of (X, y_dict) where:
            - X: Features for model training
            - y_dict: Dictionary with 'wins', 'margin', 'total' targets
            
        Example:
            X, y = engineer.split_features_targets(features_df)
            X_train, X_test, y_train, y_test = train_test_split(X, y['wins'])
        """
        feature_cols = self.get_feature_importance_names(features_df)
        
        X = features_df[feature_cols]
        
        # Only create targets if they exist (completed games)
        y_dict = {}
        if 'home_wins' in features_df.columns:
            y_dict['wins'] = features_df['home_wins']
        if 'margin' in features_df.columns:
            y_dict['margin'] = features_df['margin']
        if 'total_points' in features_df.columns:
            y_dict['total'] = features_df['total_points']
        
        return X, y_dict
    
    def create_matchup_summary(self,
                              away_team: str,
                              home_team: str,
                              features: pd.Series) -> str:
        """
        Create human-readable summary of key matchup features
        
        Args:
            away_team: Away team abbreviation
            home_team: Home team abbreviation
            features: Feature row for this game
            
        Returns:
            Formatted string with key matchup insights
            
        Example:
            summary = engineer.create_matchup_summary('BUF', 'KC', features)
            print(summary)
        """
        summary_lines = [
            f"\n{'='*60}",
            f"MATCHUP: {away_team} @ {home_team}",
            f"{'='*60}",
            "\nKEY DIFFERENTIALS (positive favors home team):"
        ]
        
        # Key statistical differentials
        key_diffs = {
            'diff_total_epa': 'EPA',
            'diff_total_offense': 'Total Offense (yards)',
            'diff_total_tds': 'Touchdowns',
            'diff_offensive_efficiency': 'Offensive Efficiency',
            'diff_defensive_efficiency': 'Defensive Efficiency'
        }
        
        for feat, label in key_diffs.items():
            if feat in features.index:
                value = features[feat]
                if abs(value) > 0.1:  # Only show meaningful differences
                    direction = home_team if value > 0 else away_team
                    summary_lines.append(f"  {label}: {value:+.2f} (favors {direction})")
        
        # Recent form comparison
        summary_lines.append("\nRECENT FORM (last 4 games):")
        
        if 'away_recent_wins' in features.index:
            away_record = f"{int(features['away_recent_wins'])}-{int(features['away_recent_wins'] - features['away_recent_wins'] + 4)}"
            home_record = f"{int(features['home_recent_wins'])}-{int(features['home_recent_wins'] - features['home_recent_wins'] + 4)}"
            
            summary_lines.append(f"  {away_team}: {away_record} record, {features['away_recent_avg_margin']:+.1f} avg margin")
            summary_lines.append(f"  {home_team}: {home_record} record, {features['home_recent_avg_margin']:+.1f} avg margin")
        
        summary_lines.append(f"\n{'='*60}")
        
        return '\n'.join(summary_lines)


# Convenience function
def engineer_features(schedules: pd.DataFrame,
                     team_averages: pd.DataFrame,
                     recent_form: pd.DataFrame) -> pd.DataFrame:
    """
    Quick function to engineer game features
    
    Args:
        schedules: Game schedules
        team_averages: Team statistical averages
        recent_form: Recent form data
        
    Returns:
        Engineered features DataFrame
        
    Example:
        from src.features.engineer import engineer_features
        features = engineer_features(schedules, team_averages, recent_form)
    """
    engineer = FeatureEngineer()
    return engineer.create_game_features(schedules, team_averages, recent_form)