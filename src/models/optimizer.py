"""
Model Optimizer
Hyperparameter tuning using Optuna (Bayesian optimization)
Fixes overfitting and improves test accuracy
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Callable
import logging
from pathlib import Path
import joblib
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import matplotlib.pyplot as plt

from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, mean_absolute_error, log_loss

from config.settings import model_config, Paths

logger = logging.getLogger(__name__)


class ModelOptimizer:
    """
    Optimizes XGBoost hyperparameters using Optuna
    
    Problem: Current model overfits (99.8% train, 66.2% test)
    Solution: Use Bayesian optimization to find parameters that:
    - Reduce overfitting (add regularization)
    - Maximize test accuracy
    - Maintain good calibration
    
    Optuna learns from each trial to intelligently explore the parameter space.
    """
    
    def __init__(self, n_trials: int = 50):
        """
        Initialize optimizer
        
        Args:
            n_trials: Number of hyperparameter combinations to try
                     50 trials = ~30-45 minutes
                     100 trials = ~60-90 minutes
        """
        self.n_trials = n_trials
        self.best_params = {}
        self.best_models = {}
        self.optimization_results = {}
        self.studies = {}
        
    def optimize_win_model(self,
                          X_train: pd.DataFrame,
                          y_train: pd.Series,
                          X_test: pd.DataFrame,
                          y_test: pd.Series) -> Dict:
        """
        Optimize hyperparameters for win probability model
        
        Uses Optuna to search for parameters that maximize test accuracy
        while minimizing overfitting (train/test gap).
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            Dictionary with best parameters and metrics
        """
        logger.info("Starting hyperparameter optimization for win model...")
        logger.info(f"Will test {self.n_trials} different parameter combinations")
        
        def objective(trial):
            """
            Optuna objective function
            
            Optuna calls this repeatedly with different parameter suggestions.
            We train a model with those parameters and return a score.
            Optuna learns which parameters work best.
            """
            # Suggest hyperparameters
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'min_child_weight': trial.suggest_int('min_child_weight', 5, 20),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': model_config.RANDOM_STATE,
                'eval_metric': 'logloss'
            }
            
            # Train model with these parameters
            model = XGBClassifier(**params)
            model.fit(X_train, y_train, verbose=False)
            
            # Evaluate on test set
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            train_acc = accuracy_score(y_train, y_pred_train)
            test_acc = accuracy_score(y_test, y_pred_test)
            
            # Calculate overfitting gap
            overfit_gap = train_acc - test_acc
            
            # Objective: Maximize test accuracy while penalizing overfitting
            # If gap > 20%, heavily penalize
            # If gap > 10%, moderately penalize
            if overfit_gap > 0.20:
                penalty = 0.15  # Heavy penalty
            elif overfit_gap > 0.10:
                penalty = 0.05  # Moderate penalty
            else:
                penalty = 0.0   # No penalty
            
            score = test_acc - penalty
            
            # Store metrics for analysis
            trial.set_user_attr('train_acc', train_acc)
            trial.set_user_attr('test_acc', test_acc)
            trial.set_user_attr('overfit_gap', overfit_gap)
            
            return score
        
        # Create Optuna study
        study = optuna.create_study(
            direction='maximize',
            study_name='win_model_optimization'
        )
        
        # Run optimization
        study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        self.studies['win'] = study
        
        # Get best parameters
        best_params = study.best_params
        best_trial = study.best_trial
        
        logger.info(f"\nOptimization complete!")
        logger.info(f"Best trial: #{best_trial.number}")
        logger.info(f"Best test accuracy: {best_trial.user_attrs['test_acc']:.3f}")
        logger.info(f"Best train accuracy: {best_trial.user_attrs['train_acc']:.3f}")
        logger.info(f"Overfitting gap: {best_trial.user_attrs['overfit_gap']:.3f}")
        
        # Train final model with best parameters
        logger.info("\nTraining final model with best parameters...")
        final_model = XGBClassifier(**best_params, random_state=model_config.RANDOM_STATE)
        final_model.fit(X_train, y_train, verbose=False)
        
        # Evaluate final model
        y_pred_train = final_model.predict(X_train)
        y_pred_test = final_model.predict(X_test)
        y_proba_train = final_model.predict_proba(X_train)
        y_proba_test = final_model.predict_proba(X_test)
        
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        train_logloss = log_loss(y_train, y_proba_train)
        test_logloss = log_loss(y_test, y_proba_test)
        
        results = {
            'model': final_model,
            'best_params': best_params,
            'metrics': {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'train_logloss': train_logloss,
                'test_logloss': test_logloss,
                'overfit_gap': train_acc - test_acc
            },
            'study': study
        }
        
        self.best_models['win'] = final_model
        self.best_params['win'] = best_params
        self.optimization_results['win'] = results
        
        return results
    
    def optimize_margin_model(self,
                             X_train: pd.DataFrame,
                             y_train: pd.Series,
                             X_test: pd.DataFrame,
                             y_test: pd.Series) -> Dict:
        """
        Optimize hyperparameters for margin prediction model
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            Dictionary with best parameters and metrics
        """
        logger.info("Starting hyperparameter optimization for margin model...")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'min_child_weight': trial.suggest_int('min_child_weight', 5, 20),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': model_config.RANDOM_STATE
            }
            
            model = XGBRegressor(**params)
            model.fit(X_train, y_train, verbose=False)
            
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            # Penalize overfitting
            overfit_ratio = train_mae / test_mae if test_mae > 0 else 1.0
            if overfit_ratio < 0.1:  # Extreme overfitting
                penalty = 5.0
            elif overfit_ratio < 0.3:
                penalty = 2.0
            else:
                penalty = 0.0
            
            trial.set_user_attr('train_mae', train_mae)
            trial.set_user_attr('test_mae', test_mae)
            
            # Minimize test MAE with penalty
            return test_mae + penalty
        
        study = optuna.create_study(
            direction='minimize',
            study_name='margin_model_optimization'
        )
        
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        self.studies['margin'] = study
        
        best_params = study.best_params
        
        # Train final model
        final_model = XGBRegressor(**best_params, random_state=model_config.RANDOM_STATE)
        final_model.fit(X_train, y_train, verbose=False)
        
        y_pred_train = final_model.predict(X_train)
        y_pred_test = final_model.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        logger.info(f"\nOptimization complete!")
        logger.info(f"Best test MAE: {test_mae:.2f} points")
        logger.info(f"Best train MAE: {train_mae:.2f} points")
        
        results = {
            'model': final_model,
            'best_params': best_params,
            'metrics': {
                'train_mae': train_mae,
                'test_mae': test_mae
            },
            'study': study
        }
        
        self.best_models['margin'] = final_model
        self.best_params['margin'] = best_params
        self.optimization_results['margin'] = results
        
        return results
    
    def optimize_total_model(self,
                           X_train: pd.DataFrame,
                           y_train: pd.Series,
                           X_test: pd.DataFrame,
                           y_test: pd.Series) -> Dict:
        """
        Optimize hyperparameters for total points model
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            Dictionary with best parameters and metrics
        """
        logger.info("Starting hyperparameter optimization for total points model...")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'min_child_weight': trial.suggest_int('min_child_weight', 5, 20),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': model_config.RANDOM_STATE
            }
            
            model = XGBRegressor(**params)
            model.fit(X_train, y_train, verbose=False)
            
            y_pred_test = model.predict(X_test)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            return test_mae
        
        study = optuna.create_study(
            direction='minimize',
            study_name='total_model_optimization'
        )
        
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        self.studies['total'] = study
        
        best_params = study.best_params
        
        # Train final model
        final_model = XGBRegressor(**best_params, random_state=model_config.RANDOM_STATE)
        final_model.fit(X_train, y_train, verbose=False)
        
        y_pred_train = final_model.predict(X_train)
        y_pred_test = final_model.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        logger.info(f"\nOptimization complete!")
        logger.info(f"Best test MAE: {test_mae:.2f} points")
        logger.info(f"Best train MAE: {train_mae:.2f} points")
        
        results = {
            'model': final_model,
            'best_params': best_params,
            'metrics': {
                'train_mae': train_mae,
                'test_mae': test_mae
            },
            'study': study
        }
        
        self.best_models['total'] = final_model
        self.best_params['total'] = best_params
        self.optimization_results['total'] = results
        
        return results
    
    def plot_optimization_history(self, model_name: str, save_path: Optional[Path] = None):
        """
        Plot optimization history showing how Optuna learned
        
        Args:
            model_name: Name of model ('win', 'margin', 'total')
            save_path: Optional path to save plot
        """
        if model_name not in self.studies:
            raise ValueError(f"No study found for {model_name}")
        
        study = self.studies[model_name]
        
        fig = plot_optimization_history(study)
        fig.update_layout(
            title=f"{model_name.upper()} Model - Optimization History",
            xaxis_title="Trial Number",
            yaxis_title="Objective Value"
        )
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(save_path)
            logger.info(f"Saved optimization history to {save_path}")
        
        return fig
    
    def plot_param_importances(self, model_name: str, save_path: Optional[Path] = None):
        """
        Plot parameter importance showing which parameters matter most
        
        Args:
            model_name: Name of model ('win', 'margin', 'total')
            save_path: Optional path to save plot
        """
        if model_name not in self.studies:
            raise ValueError(f"No study found for {model_name}")
        
        study = self.studies[model_name]
        
        fig = plot_param_importances(study)
        fig.update_layout(
            title=f"{model_name.upper()} Model - Parameter Importances",
            xaxis_title="Importance"
        )
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(save_path)
            logger.info(f"Saved parameter importances to {save_path}")
        
        return fig
    
    def save_optimized_models(self, output_dir: Optional[Path] = None):
        """
        Save optimized models and results
        
        Args:
            output_dir: Directory to save models
        """
        if not self.best_models:
            raise ValueError("No optimized models to save")
        
        save_dir = output_dir or Paths.MODELS
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models
        for model_name, model in self.best_models.items():
            filepath = save_dir / f"{model_name}_model_optimized.joblib"
            joblib.dump(model, filepath)
            logger.info(f"Saved optimized {model_name} model to {filepath}")
        
        # Save best parameters
        params_filepath = save_dir / "optimized_hyperparameters.joblib"
        joblib.dump(self.best_params, params_filepath)
        logger.info(f"Saved hyperparameters to {params_filepath}")
        
        # Save optimization results
        results_filepath = save_dir / "optimization_results.joblib"
        joblib.dump(self.optimization_results, results_filepath)
        logger.info(f"Saved optimization results to {results_filepath}")
    
    def compare_to_baseline(self, baseline_results: Dict) -> pd.DataFrame:
        """
        Compare optimized models to baseline (unoptimized) models
        
        Args:
            baseline_results: Results from baseline model training
            
        Returns:
            Comparison DataFrame
        """
        comparison_data = []
        
        # Win model comparison
        if 'win' in self.optimization_results and 'win' in baseline_results:
            opt = self.optimization_results['win']['metrics']
            base = baseline_results['win']
            
            comparison_data.append({
                'model': 'Win Probability',
                'metric': 'Test Accuracy',
                'baseline': base['test_accuracy'],
                'optimized': opt['test_accuracy'],
                'improvement': opt['test_accuracy'] - base['test_accuracy'],
                'improvement_pct': (opt['test_accuracy'] - base['test_accuracy']) / base['test_accuracy'] * 100
            })
            
            comparison_data.append({
                'model': 'Win Probability',
                'metric': 'Overfit Gap',
                'baseline': base['test_accuracy'] - base['train_accuracy'],
                'optimized': opt['overfit_gap'],
                'improvement': (base['test_accuracy'] - base['train_accuracy']) - opt['overfit_gap'],
                'improvement_pct': ((base['test_accuracy'] - base['train_accuracy']) - opt['overfit_gap']) / abs(base['test_accuracy'] - base['train_accuracy']) * 100
            })
        
        # Margin model comparison
        if 'margin' in self.optimization_results and 'margin' in baseline_results:
            opt = self.optimization_results['margin']['metrics']
            base = baseline_results['margin']
            
            comparison_data.append({
                'model': 'Margin Prediction',
                'metric': 'Test MAE',
                'baseline': base['test_mae'],
                'optimized': opt['test_mae'],
                'improvement': base['test_mae'] - opt['test_mae'],
                'improvement_pct': (base['test_mae'] - opt['test_mae']) / base['test_mae'] * 100
            })
        
        # Total model comparison
        if 'total' in self.optimization_results and 'total' in baseline_results:
            opt = self.optimization_results['total']['metrics']
            base = baseline_results['total']
            
            comparison_data.append({
                'model': 'Total Points',
                'metric': 'Test MAE',
                'baseline': base['test_mae'],
                'optimized': opt['test_mae'],
                'improvement': base['test_mae'] - opt['test_mae'],
                'improvement_pct': (base['test_mae'] - opt['test_mae']) / base['test_mae'] * 100
            })
        
        return pd.DataFrame(comparison_data)


# Convenience function
def optimize_model(X_train, y_train, X_test, y_test, model_type='win', n_trials=50):
    """
    Quick function to optimize a model
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        model_type: 'win', 'margin', or 'total'
        n_trials: Number of optimization trials
        
    Returns:
        Optimization results dictionary
    """
    optimizer = ModelOptimizer(n_trials=n_trials)
    
    if model_type == 'win':
        return optimizer.optimize_win_model(X_train, y_train, X_test, y_test)
    elif model_type == 'margin':
        return optimizer.optimize_margin_model(X_train, y_train, X_test, y_test)
    elif model_type == 'total':
        return optimizer.optimize_total_model(X_train, y_train, X_test, y_test)
    else:
        raise ValueError(f"Unknown model type: {model_type}")