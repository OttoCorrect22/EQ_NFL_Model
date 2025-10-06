"""
Model Calibrator
Calibrates probability predictions using isotonic regression
Fixes overconfident predictions to match reality
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import logging
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

from config.settings import Paths

logger = logging.getLogger(__name__)


class ModelCalibrator:
    """
    Calibrates XGBoost probability predictions using isotonic regression
    
    Problem: Model says "70% win probability" but teams only win 60% of the time
    Solution: Learn a calibration mapping that corrects these probabilities
    
    Research shows calibration-optimized models have 69.86% better ROI than
    accuracy-optimized models.
    """
    
    def __init__(self):
        """Initialize calibrator"""
        self.calibrated_models = {}
        self.calibration_metrics = {}
        
    def calibrate_classifier(self,
                            model,
                            X_train: pd.DataFrame,
                            y_train: pd.Series,
                            X_test: pd.DataFrame,
                            y_test: pd.Series,
                            model_name: str = 'model') -> CalibratedClassifierCV:
        """
        Calibrate a classifier using isotonic regression
        
        Isotonic regression learns a monotonic (always increasing) function that
        maps uncalibrated probabilities to calibrated ones.
        
        Args:
            model: Trained classifier (e.g., XGBoost)
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            model_name: Name for logging
            
        Returns:
            Calibrated classifier
            
        Example:
            calibrator = ModelCalibrator()
            calibrated_win_model = calibrator.calibrate_classifier(
                win_model, X_train, y_train, X_test, y_test, 'win'
            )
        """
        logger.info(f"Calibrating {model_name} model using isotonic regression...")
        
        # Get uncalibrated predictions
        y_train_proba_uncal = model.predict_proba(X_train)[:, 1]
        y_test_proba_uncal = model.predict_proba(X_test)[:, 1]
        
        # Calculate uncalibrated metrics
        uncal_train_logloss = log_loss(y_train, y_train_proba_uncal)
        uncal_test_logloss = log_loss(y_test, y_test_proba_uncal)
        uncal_train_brier = brier_score_loss(y_train, y_train_proba_uncal)
        uncal_test_brier = brier_score_loss(y_test, y_test_proba_uncal)
        
        logger.info(f"Uncalibrated {model_name} - Log Loss: {uncal_test_logloss:.3f}, Brier: {uncal_test_brier:.3f}")
        
        # Calibrate using isotonic regression
        # method='isotonic': Non-parametric, learns arbitrary monotonic function
        # cv='prefit': Don't retrain the base model, just calibrate probabilities
        calibrated_model = CalibratedClassifierCV(
            model,
            method='isotonic',
            cv='prefit'
        )
        
        # Fit calibration on training data
        calibrated_model.fit(X_train, y_train)
        
        # Get calibrated predictions
        y_train_proba_cal = calibrated_model.predict_proba(X_train)[:, 1]
        y_test_proba_cal = calibrated_model.predict_proba(X_test)[:, 1]
        
        # Calculate calibrated metrics
        cal_train_logloss = log_loss(y_train, y_train_proba_cal)
        cal_test_logloss = log_loss(y_test, y_test_proba_cal)
        cal_train_brier = brier_score_loss(y_train, y_train_proba_cal)
        cal_test_brier = brier_score_loss(y_test, y_test_proba_cal)
        
        logger.info(f"Calibrated {model_name} - Log Loss: {cal_test_logloss:.3f}, Brier: {cal_test_brier:.3f}")
        
        # Calculate improvements
        logloss_improvement = (uncal_test_logloss - cal_test_logloss) / uncal_test_logloss * 100
        brier_improvement = (uncal_test_brier - cal_test_brier) / uncal_test_brier * 100
        
        logger.info(f"Improvements: Log Loss {logloss_improvement:+.1f}%, Brier {brier_improvement:+.1f}%")
        
        # Store metrics
        self.calibration_metrics[model_name] = {
            'uncalibrated': {
                'train_logloss': uncal_train_logloss,
                'test_logloss': uncal_test_logloss,
                'train_brier': uncal_train_brier,
                'test_brier': uncal_test_brier
            },
            'calibrated': {
                'train_logloss': cal_train_logloss,
                'test_logloss': cal_test_logloss,
                'train_brier': cal_train_brier,
                'test_brier': cal_test_brier
            },
            'improvements': {
                'logloss_improvement_pct': logloss_improvement,
                'brier_improvement_pct': brier_improvement
            }
        }
        
        self.calibrated_models[model_name] = calibrated_model
        
        return calibrated_model
    
    def plot_calibration_curve(self,
                               model,
                               calibrated_model,
                               X_test: pd.DataFrame,
                               y_test: pd.Series,
                               model_name: str = 'model',
                               save_path: Optional[Path] = None):
        """
        Create calibration curve comparing uncalibrated vs calibrated probabilities
        
        Perfect calibration: When model predicts 70%, actual win rate is 70%
        Overconfident: Model predicts 70%, actual win rate is 55%
        Underconfident: Model predicts 70%, actual win rate is 85%
        
        Args:
            model: Uncalibrated model
            calibrated_model: Calibrated model
            X_test: Test features
            y_test: Test labels
            model_name: Name for plot title
            save_path: Optional path to save plot
        """
        # Get predictions
        y_proba_uncal = model.predict_proba(X_test)[:, 1]
        y_proba_cal = calibrated_model.predict_proba(X_test)[:, 1]
        
        # Calculate calibration curves
        fraction_of_positives_uncal, mean_predicted_value_uncal = calibration_curve(
            y_test, y_proba_uncal, n_bins=10, strategy='uniform'
        )
        fraction_of_positives_cal, mean_predicted_value_cal = calibration_curve(
            y_test, y_proba_cal, n_bins=10, strategy='uniform'
        )
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Calibration curves
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
        ax1.plot(mean_predicted_value_uncal, fraction_of_positives_uncal, 
                'ro-', label='Uncalibrated', linewidth=2, markersize=8)
        ax1.plot(mean_predicted_value_cal, fraction_of_positives_cal,
                'go-', label='Calibrated (Isotonic)', linewidth=2, markersize=8)
        
        ax1.set_xlabel('Predicted Probability', fontsize=12)
        ax1.set_ylabel('Actual Win Rate', fontsize=12)
        ax1.set_title(f'{model_name.upper()} - Calibration Curve', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        
        # Plot 2: Prediction distribution
        ax2.hist(y_proba_uncal, bins=20, alpha=0.5, label='Uncalibrated', color='red')
        ax2.hist(y_proba_cal, bins=20, alpha=0.5, label='Calibrated', color='green')
        ax2.set_xlabel('Predicted Probability', fontsize=12)
        ax2.set_ylabel('Number of Predictions', fontsize=12)
        ax2.set_title(f'{model_name.upper()} - Probability Distribution', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved calibration plot to {save_path}")
        
        plt.show()
        
        return fig
    
    def save_calibrated_models(self, output_dir: Optional[Path] = None):
        """
        Save calibrated models to disk
        
        Args:
            output_dir: Directory to save models (uses config default if None)
        """
        if not self.calibrated_models:
            raise ValueError("No calibrated models to save. Calibrate models first.")
        
        save_dir = output_dir or Paths.MODELS
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.calibrated_models.items():
            filepath = save_dir / f"{model_name}_model_calibrated.joblib"
            joblib.dump(model, filepath)
            logger.info(f"Saved calibrated {model_name} model to {filepath}")
        
        # Save calibration metrics
        metrics_filepath = save_dir / "calibration_metrics.joblib"
        joblib.dump(self.calibration_metrics, metrics_filepath)
        logger.info(f"Saved calibration metrics to {metrics_filepath}")
    
    def load_calibrated_models(self, model_dir: Optional[Path] = None):
        """
        Load calibrated models from disk
        
        Args:
            model_dir: Directory containing saved models
        """
        load_dir = model_dir or Paths.MODELS
        
        self.calibrated_models = {}
        for model_name in ['win']:  # Only win model needs calibration
            filepath = load_dir / f"{model_name}_model_calibrated.joblib"
            if filepath.exists():
                self.calibrated_models[model_name] = joblib.load(filepath)
                logger.info(f"Loaded calibrated {model_name} model from {filepath}")
            else:
                logger.warning(f"Calibrated model file not found: {filepath}")
        
        # Load metrics
        metrics_filepath = load_dir / "calibration_metrics.joblib"
        if metrics_filepath.exists():
            self.calibration_metrics = joblib.load(metrics_filepath)
            logger.info(f"Loaded calibration metrics from {metrics_filepath}")
    
    def compare_predictions(self,
                           model,
                           calibrated_model,
                           X_sample: pd.DataFrame,
                           n_samples: int = 10) -> pd.DataFrame:
        """
        Compare uncalibrated vs calibrated predictions on sample data
        
        Shows how calibration fixes overconfident predictions
        
        Args:
            model: Uncalibrated model
            calibrated_model: Calibrated model
            X_sample: Sample features
            n_samples: Number of samples to compare
            
        Returns:
            DataFrame with comparison
        """
        # Get predictions
        uncal_proba = model.predict_proba(X_sample[:n_samples])[:, 1]
        cal_proba = calibrated_model.predict_proba(X_sample[:n_samples])[:, 1]
        
        # Create comparison DataFrame
        comparison = pd.DataFrame({
            'sample': range(1, n_samples + 1),
            'uncalibrated_prob': uncal_proba,
            'calibrated_prob': cal_proba,
            'difference': cal_proba - uncal_proba,
            'adjustment_pct': ((cal_proba - uncal_proba) / uncal_proba * 100)
        })
        
        return comparison
    
    def get_metrics_summary(self) -> pd.DataFrame:
        """
        Get summary of calibration metrics for all models
        
        Returns:
            DataFrame with metrics comparison
        """
        if not self.calibration_metrics:
            raise ValueError("No calibration metrics available")
        
        summary_data = []
        
        for model_name, metrics in self.calibration_metrics.items():
            summary_data.append({
                'model': model_name,
                'uncal_test_logloss': metrics['uncalibrated']['test_logloss'],
                'cal_test_logloss': metrics['calibrated']['test_logloss'],
                'logloss_improvement': metrics['improvements']['logloss_improvement_pct'],
                'uncal_test_brier': metrics['uncalibrated']['test_brier'],
                'cal_test_brier': metrics['calibrated']['test_brier'],
                'brier_improvement': metrics['improvements']['brier_improvement_pct']
            })
        
        return pd.DataFrame(summary_data)


# Convenience function
def calibrate_model(model, X_train, y_train, X_test, y_test, model_name='model'):
    """
    Quick function to calibrate a model
    
    Args:
        model: Trained model
        X_train, y_train: Training data
        X_test, y_test: Test data
        model_name: Name for the model
        
    Returns:
        Calibrated model
        
    Example:
        from src.models.calibrator import calibrate_model
        calibrated_win_model = calibrate_model(
            win_model, X_train, y_train, X_test, y_test, 'win'
        )
    """
    calibrator = ModelCalibrator()
    return calibrator.calibrate_classifier(
        model, X_train, y_train, X_test, y_test, model_name
    )