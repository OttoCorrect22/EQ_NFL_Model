"""
Model Trainer
Trains XGBoost models for NFL game prediction with walk-forward validation
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging
from pathlib import Path
import joblib

from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, log_loss
import shap

from config.settings import model_config, Paths

logger = logging.getLogger(__name__)


class NFLModelTrainer:
    """
    Trains XGBoost models for NFL prediction
    
    Implements walk-forward validation to avoid temporal data leakage.
    Trains three models:
    1. Win probability (classification)
    2. Margin prediction (regression)
    3. Total points prediction (regression)
    
    Based on research showing XGBoost achieves 60-65% win accuracy.
    """
    
    def __init__(self):
        """Initialize trainer with configuration"""
        self.config = model_config
        self.models = {}
        self.feature_names = []
        self.performance_metrics = {}
        
    def train_all_models(self, features_df: pd.DataFrame) -> Dict:
        """
        Train all three models with walk-forward validation
        
        Args:
            features_df: DataFrame with engineered features
                        Must have: season, home_wins, margin, total_points
                        
        Returns:
            Dictionary with trained models and performance metrics
            
        Example:
            trainer = NFLModelTrainer()
            results = trainer.train_all_models(features_df)
            win_model = results['models']['win']
        """
        logger.info("Starting model training with walk-forward validation")
        
        # Split by season for walk-forward validation
        train_df, test_df = self._walk_forward_split(features_df)
        
        logger.info(f"Training set: {len(train_df)} games (seasons {train_df['season'].min()}-{train_df['season'].max()})")
        logger.info(f"Test set: {len(test_df)} games (season {test_df['season'].min()})")
        
        # Prepare features and targets
        X_train, y_train = self._prepare_data(train_df)
        X_test, y_test = self._prepare_data(test_df)
        
        self.feature_names = X_train.columns.tolist()
        
        # Train win probability model
        logger.info("\n" + "="*60)
        logger.info("Training Win Probability Model (XGBoost Classifier)")
        logger.info("="*60)
        self.models['win'] = self._train_win_model(
            X_train, y_train['wins'], X_test, y_test['wins']
        )
        
        # Train margin model
        logger.info("\n" + "="*60)
        logger.info("Training Margin Model (XGBoost Regressor)")
        logger.info("="*60)
        self.models['margin'] = self._train_margin_model(
            X_train, y_train['margin'], X_test, y_test['margin']
        )
        
        # Train total points model
        logger.info("\n" + "="*60)
        logger.info("Training Total Points Model (XGBoost Regressor)")
        logger.info("="*60)
        self.models['total'] = self._train_total_model(
            X_train, y_train['total'], X_test, y_test['total']
        )
        
        return {
            'models': self.models,
            'metrics': self.performance_metrics,
            'feature_names': self.feature_names
        }
    
    def _walk_forward_split(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data using walk-forward validation
        
        Train on earlier seasons, test on most recent season.
        This mimics real prediction scenario.
        
        Args:
            features_df: Full features DataFrame
            
        Returns:
            Tuple of (train_df, test_df)
        """
        # Get the most recent season for testing
        test_season = features_df['season'].max()
        
        # All earlier seasons for training
        train_df = features_df[features_df['season'] < test_season].copy()
        test_df = features_df[features_df['season'] == test_season].copy()
        
        return train_df, test_df
    
    def _prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """
        Prepare features and targets from DataFrame
        
        Args:
            df: Features DataFrame
            
        Returns:
            Tuple of (X, y_dict) where y_dict has 'wins', 'margin', 'total'
        """
        # Exclude identifier and target columns
        exclude_cols = [
            'game_id', 'season', 'week', 'away_team', 'home_team',
            'home_score', 'away_score', 'home_wins', 'margin', 'total_points'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        
        y_dict = {
            'wins': df['home_wins'].copy(),
            'margin': df['margin'].copy(),
            'total': df['total_points'].copy()
        }
        
        return X, y_dict
    
    def _train_win_model(self, 
                        X_train: pd.DataFrame, 
                        y_train: pd.Series,
                        X_test: pd.DataFrame,
                        y_test: pd.Series) -> XGBClassifier:
        """
        Train XGBoost classifier for win probability
        
        Uses hyperparameters recommended by research for NFL prediction
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            Trained XGBClassifier
        """
        # Hyperparameters from research document
        model = XGBClassifier(
            n_estimators=500,
            learning_rate=0.1,
            max_depth=7,
            min_child_weight=6,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=self.config.RANDOM_STATE,
            eval_metric='logloss'
        )
        
        # Train
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Evaluate
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        # Probability predictions for log loss
        train_proba = model.predict_proba(X_train)
        test_proba = model.predict_proba(X_test)
        
        train_logloss = log_loss(y_train, train_proba)
        test_logloss = log_loss(y_test, test_proba)
        
        logger.info(f"Training accuracy: {train_acc:.3f}")
        logger.info(f"Test accuracy: {test_acc:.3f}")
        logger.info(f"Training log loss: {train_logloss:.3f}")
        logger.info(f"Test log loss: {test_logloss:.3f}")
        
        # Store metrics
        self.performance_metrics['win'] = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'train_logloss': train_logloss,
            'test_logloss': test_logloss
        }
        
        # Feature importance
        self._log_feature_importance(model, 'Win Model')
        
        return model
    
    def _train_margin_model(self,
                          X_train: pd.DataFrame,
                          y_train: pd.Series,
                          X_test: pd.DataFrame,
                          y_test: pd.Series) -> XGBRegressor:
        """
        Train XGBoost regressor for margin prediction
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            Trained XGBRegressor
        """
        model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.1,
            max_depth=7,
            min_child_weight=6,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=self.config.RANDOM_STATE,
            objective='reg:squarederror'
        )
        
        # Train
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Evaluate
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        logger.info(f"Training MAE: {train_mae:.2f} points")
        logger.info(f"Test MAE: {test_mae:.2f} points")
        
        # Store metrics
        self.performance_metrics['margin'] = {
            'train_mae': train_mae,
            'test_mae': test_mae
        }
        
        # Feature importance
        self._log_feature_importance(model, 'Margin Model')
        
        return model
    
    def _train_total_model(self,
                         X_train: pd.DataFrame,
                         y_train: pd.Series,
                         X_test: pd.DataFrame,
                         y_test: pd.Series) -> XGBRegressor:
        """
        Train XGBoost regressor for total points prediction
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            Trained XGBRegressor
        """
        model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.1,
            max_depth=7,
            min_child_weight=6,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=self.config.RANDOM_STATE,
            objective='reg:squarederror'
        )
        
        # Train
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Evaluate
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        logger.info(f"Training MAE: {train_mae:.2f} points")
        logger.info(f"Test MAE: {test_mae:.2f} points")
        
        # Store metrics
        self.performance_metrics['total'] = {
            'train_mae': train_mae,
            'test_mae': test_mae
        }
        
        # Feature importance
        self._log_feature_importance(model, 'Total Points Model')
        
        return model
    
    def _log_feature_importance(self, model, model_name: str, top_n: int = 10):
        """
        Log top N most important features
        
        Args:
            model: Trained model
            model_name: Name for logging
            top_n: Number of top features to show
        """
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"\nTop {top_n} features for {model_name}:")
        for idx, row in importance_df.head(top_n).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    def save_models(self, output_dir: Optional[Path] = None):
        """
        Save trained models to disk
        
        Args:
            output_dir: Directory to save models (uses config default if None)
        """
        if not self.models:
            raise ValueError("No models to save. Train models first.")
        
        save_dir = output_dir or Paths.MODELS
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.models.items():
            filepath = save_dir / f"{model_name}_model.joblib"
            joblib.dump(model, filepath)
            logger.info(f"Saved {model_name} model to {filepath}")
        
        # Save feature names
        feature_filepath = save_dir / "feature_names.joblib"
        joblib.dump(self.feature_names, feature_filepath)
        logger.info(f"Saved feature names to {feature_filepath}")
        
        # Save performance metrics
        metrics_filepath = save_dir / "performance_metrics.joblib"
        joblib.dump(self.performance_metrics, metrics_filepath)
        logger.info(f"Saved performance metrics to {metrics_filepath}")
    
    def load_models(self, model_dir: Optional[Path] = None):
        """
        Load trained models from disk
        
        Args:
            model_dir: Directory containing saved models
        """
        load_dir = model_dir or Paths.MODELS
        
        self.models = {}
        for model_name in ['win', 'margin', 'total']:
            filepath = load_dir / f"{model_name}_model.joblib"
            if filepath.exists():
                self.models[model_name] = joblib.load(filepath)
                logger.info(f"Loaded {model_name} model from {filepath}")
            else:
                logger.warning(f"Model file not found: {filepath}")
        
        # Load feature names
        feature_filepath = load_dir / "feature_names.joblib"
        if feature_filepath.exists():
            self.feature_names = joblib.load(feature_filepath)
            logger.info(f"Loaded feature names from {feature_filepath}")
        
        # Load metrics
        metrics_filepath = load_dir / "performance_metrics.joblib"
        if metrics_filepath.exists():
            self.performance_metrics = joblib.load(metrics_filepath)
            logger.info(f"Loaded performance metrics from {metrics_filepath}")


# Convenience function
def train_models(features_df: pd.DataFrame) -> NFLModelTrainer:
    """
    Quick function to train all models
    
    Args:
        features_df: Engineered features DataFrame
        
    Returns:
        Trained NFLModelTrainer instance
        
    Example:
        from src.models.trainer import train_models
        trainer = train_models(features_df)
        trainer.save_models()
    """
    trainer = NFLModelTrainer()
    trainer.train_all_models(features_df)
    return trainer