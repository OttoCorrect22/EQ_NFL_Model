"""
Model Predictor
Generates predictions for NFL games using trained models
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging
from pathlib import Path

from config.settings import Paths

logger = logging.getLogger(__name__)


class NFLPredictor:
    """
    Generates predictions using trained models
    
    Takes game features and uses the three trained models to predict:
    - Win probability
    - Point margin
    - Total points
    
    Then calculates final scores ensuring mathematical consistency.
    """
    
    def __init__(self, trainer=None):
        """
        Initialize predictor
        
        Args:
            trainer: NFLModelTrainer instance with trained models
                    If None, must load models separately
        """
        if trainer is not None:
            self.models = trainer.models
            self.feature_names = trainer.feature_names
        else:
            self.models = {}
            self.feature_names = []
    
    def load_models(self, model_dir: Optional[Path] = None):
        """
        Load trained models from disk
        
        Args:
            model_dir: Directory containing saved models
        """
        import joblib
        
        load_dir = model_dir or Paths.MODELS
        
        self.models = {}
        for model_name in ['win', 'margin', 'total']:
            filepath = load_dir / f"{model_name}_model.joblib"
            if filepath.exists():
                self.models[model_name] = joblib.load(filepath)
                logger.info(f"Loaded {model_name} model")
            else:
                raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load feature names
        feature_filepath = load_dir / "feature_names.joblib"
        if feature_filepath.exists():
            self.feature_names = joblib.load(feature_filepath)
            logger.info(f"Loaded {len(self.feature_names)} feature names")
        else:
            raise FileNotFoundError(f"Feature names file not found: {feature_filepath}")
    
    def predict_games(self, game_features: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for multiple games
        
        Args:
            game_features: DataFrame with game features
                          Must include: away_team, home_team, and all model features
                          
        Returns:
            DataFrame with predictions for each game
            
        Example:
            predictor = NFLPredictor()
            predictor.load_models()
            predictions = predictor.predict_games(upcoming_games_features)
        """
        if not self.models:
            raise ValueError("No models loaded. Call load_models() first.")
        
        logger.info(f"Generating predictions for {len(game_features)} games")
        
        predictions = []
        
        for idx, game in game_features.iterrows():
            try:
                prediction = self._predict_single_game(game)
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Error predicting game {idx}: {e}")
                continue
        
        predictions_df = pd.DataFrame(predictions)
        
        logger.info(f"Generated {len(predictions_df)} predictions")
        
        return predictions_df
    
    def _predict_single_game(self, game: pd.Series) -> Dict:
        """
        Generate prediction for a single game
        
        Args:
            game: Game features as Series
            
        Returns:
            Dictionary with all prediction outputs
        """
        # Extract identifiers
        away_team = game['away_team']
        home_team = game['home_team']
        
        # Prepare features for models
        X = self._prepare_features(game)
        
        # Get predictions from each model
        win_proba = self.models['win'].predict_proba(X)[0]
        predicted_margin = self.models['margin'].predict(X)[0]
        predicted_total = self.models['total'].predict(X)[0]
        
        # Calculate win probabilities
        home_win_prob = win_proba[1]  # Probability of home win
        away_win_prob = win_proba[0]  # Probability of away win
        
        # Calculate final scores (mathematically consistent)
        home_score, away_score = self._calculate_scores(
            predicted_margin, predicted_total
        )
        
        # Determine predicted winner
        predicted_winner = home_team if home_win_prob > 0.5 else away_team
        confidence = max(home_win_prob, away_win_prob)
        
        # Build prediction dictionary
        prediction = {
            'away_team': away_team,
            'home_team': home_team,
            'predicted_winner': predicted_winner,
            'home_win_prob': round(home_win_prob, 3),
            'away_win_prob': round(away_win_prob, 3),
            'confidence': round(confidence, 3),
            'predicted_margin': round(predicted_margin, 1),
            'predicted_total': round(predicted_total, 1),
            'home_score': int(round(home_score)),
            'away_score': int(round(away_score))
        }
        
        # Add context if available
        if 'week' in game.index:
            prediction['week'] = game['week']
        if 'season' in game.index:
            prediction['season'] = game['season']
        if 'gameday' in game.index:
            prediction['gameday'] = game['gameday']
        
        return prediction
    
    def _prepare_features(self, game: pd.Series) -> pd.DataFrame:
        """
        Prepare features for model prediction
        
        Args:
            game: Game features
            
        Returns:
            DataFrame with features in correct order
        """
        # Extract only the features used by the model
        features = {}
        for feature in self.feature_names:
            if feature in game.index:
                features[feature] = game[feature]
            else:
                # Feature missing - use 0 as default
                features[feature] = 0.0
                logger.warning(f"Feature '{feature}' missing, using 0.0")
        
        return pd.DataFrame([features])
    
    def _calculate_scores(self, margin: float, total: float) -> tuple:
        """
        Calculate individual team scores from margin and total
        
        Ensures mathematical consistency:
        home_score - away_score = margin
        home_score + away_score = total
        
        Args:
            margin: Predicted margin (positive = home favored)
            total: Predicted total points
            
        Returns:
            Tuple of (home_score, away_score)
        """
        # Solve the system of equations
        # home + away = total
        # home - away = margin
        # Therefore: home = (total + margin) / 2
        
        if margin >= 0:
            home_score = (total + margin) / 2
            away_score = (total - margin) / 2
        else:
            away_score = (total + abs(margin)) / 2
            home_score = (total - abs(margin)) / 2
        
        # Apply reasonable bounds
        home_score = max(10, min(50, home_score))
        away_score = max(10, min(50, away_score))
        
        return home_score, away_score
    
    def format_prediction_summary(self, prediction: Dict) -> str:
        """
        Format prediction as human-readable string
        
        Args:
            prediction: Prediction dictionary
            
        Returns:
            Formatted string
        """
        away = prediction['away_team']
        home = prediction['home_team']
        winner = prediction['predicted_winner']
        confidence = prediction['confidence']
        away_score = prediction['away_score']
        home_score = prediction['home_score']
        home_prob = prediction['home_win_prob']
        away_prob = prediction['away_win_prob']
        
        lines = [
            f"\n{away} @ {home}",
            f"  Winner: {winner} ({confidence:.1%} confidence)",
            f"  Win probabilities: {home} {home_prob:.1%} | {away} {away_prob:.1%}",
            f"  Predicted score: {away} {away_score} - {home_score} {home}",
            f"  Expected margin: {abs(prediction['predicted_margin']):.1f} points"
        ]
        
        return '\n'.join(lines)
    
    def save_predictions(self, 
                        predictions_df: pd.DataFrame,
                        filename: str = "predictions.csv",
                        output_dir: Optional[Path] = None):
        """
        Save predictions to file
        
        Args:
            predictions_df: Predictions DataFrame
            filename: Output filename
            output_dir: Output directory (uses config default if None)
        """
        save_dir = output_dir or Paths.PREDICTIONS
        save_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = save_dir / filename
        predictions_df.to_csv(filepath, index=False)
        
        logger.info(f"Saved {len(predictions_df)} predictions to {filepath}")
    
    def compare_to_actual(self, 
                         predictions_df: pd.DataFrame,
                         actual_results: pd.DataFrame) -> Dict:
        """
        Compare predictions to actual game results
        
        Args:
            predictions_df: Predictions DataFrame
            actual_results: DataFrame with actual scores
                          Must have: away_team, home_team, away_score, home_score
                          
        Returns:
            Dictionary with accuracy metrics
        """
        logger.info("Comparing predictions to actual results")
        
        metrics = {
            'total_games': 0,
            'correct_winners': 0,
            'win_accuracy': 0.0,
            'avg_margin_error': 0.0,
            'avg_total_error': 0.0
        }
        
        margin_errors = []
        total_errors = []
        
        for _, pred in predictions_df.iterrows():
            # Find matching actual result
            actual = actual_results[
                (actual_results['away_team'] == pred['away_team']) &
                (actual_results['home_team'] == pred['home_team'])
            ]
            
            if len(actual) == 0:
                continue
            
            actual = actual.iloc[0]
            
            # Check if scores are available
            if pd.isna(actual['away_score']) or pd.isna(actual['home_score']):
                continue
            
            metrics['total_games'] += 1
            
            # Actual results
            actual_home_score = float(actual['home_score'])
            actual_away_score = float(actual['away_score'])
            actual_winner = actual['home_team'] if actual_home_score > actual_away_score else actual['away_team']
            actual_margin = actual_home_score - actual_away_score
            actual_total = actual_home_score + actual_away_score
            
            # Check winner prediction
            if pred['predicted_winner'] == actual_winner:
                metrics['correct_winners'] += 1
            
            # Calculate errors
            margin_error = abs(pred['predicted_margin'] - actual_margin)
            total_error = abs(pred['predicted_total'] - actual_total)
            
            margin_errors.append(margin_error)
            total_errors.append(total_error)
        
        if metrics['total_games'] > 0:
            metrics['win_accuracy'] = metrics['correct_winners'] / metrics['total_games']
            metrics['avg_margin_error'] = np.mean(margin_errors)
            metrics['avg_total_error'] = np.mean(total_errors)
        
        logger.info(f"Win accuracy: {metrics['win_accuracy']:.1%}")
        logger.info(f"Average margin error: {metrics['avg_margin_error']:.1f} points")
        logger.info(f"Average total error: {metrics['avg_total_error']:.1f} points")
        
        return metrics


# Convenience function
def predict_games(game_features: pd.DataFrame, model_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Quick function to generate predictions
    
    Args:
        game_features: Game features DataFrame
        model_dir: Directory with trained models
        
    Returns:
        Predictions DataFrame
        
    Example:
        from src.models.predictor import predict_games
        predictions = predict_games(upcoming_games_features)
    """
    predictor = NFLPredictor()
    predictor.load_models(model_dir)
    return predictor.predict_games(game_features)