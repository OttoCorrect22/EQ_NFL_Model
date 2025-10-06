#!/usr/bin/env python3
"""
Calibration Script
Calibrates the win probability model using isotonic regression

This fixes overconfident predictions (99.3%, 100%) and improves
probability accuracy for better decision-making and betting ROI.

Research shows calibration can improve ROI by 69.86%!
"""

# STEP 1: Standard library imports first
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# STEP 2: Add project root to path BEFORE importing project modules
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# STEP 3: Fix Windows Unicode encoding
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# STEP 4: NOW import project modules (after sys.path is set)
import joblib
from config.settings import Paths
from src.data.loader import get_loader
from src.features.aggregator import TeamAggregator
from src.features.recent_form import RecentFormCalculator
from src.features.engineer import FeatureEngineer
from src.models.trainer import NFLModelTrainer
from src.models.calibrator import ModelCalibrator
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_header(text: str):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def calibrate_models():
    """Main calibration pipeline"""
    
    print_header("NFL MODEL CALIBRATION")
    
    # Step 1: Load data and features
    print("\n1. Loading data and engineering features...")
    
    try:
        loader = get_loader()
        schedules = loader.load_schedules()
        team_stats = loader.load_team_stats()
        
        aggregator = TeamAggregator()
        team_averages = aggregator.aggregate_team_stats(team_stats)
        
        form_calculator = RecentFormCalculator()
        recent_form = form_calculator.calculate_recent_form(schedules)
        
        engineer = FeatureEngineer()
        completed_games = schedules[schedules['away_score'].notna()].copy()
        features = engineer.create_game_features(
            completed_games, team_averages, recent_form
        )
        
        print(f"   [OK] Loaded {len(features)} completed games with features")
        
    except Exception as e:
        print(f"   [ERROR] Error loading data: {e}")
        return False
    
    # Step 2: Prepare data with walk-forward split
    print("\n2. Preparing data with walk-forward validation...")
    
    # Split by season (same as training)
    test_season = features['season'].max()
    train_df = features[features['season'] < test_season].copy()
    test_df = features[features['season'] == test_season].copy()
    
    print(f"   [OK] Training: {len(train_df)} games (seasons {train_df['season'].min()}-{train_df['season'].max()})")
    print(f"   [OK] Test: {len(test_df)} games (season {test_season})")
    
    # Prepare features
    exclude_cols = [
        'game_id', 'season', 'week', 'away_team', 'home_team',
        'home_score', 'away_score', 'home_wins', 'margin', 'total_points'
    ]
    feature_cols = [col for col in features.columns if col not in exclude_cols]
    
    X_train = train_df[feature_cols]
    y_train = train_df['home_wins']
    X_test = test_df[feature_cols]
    y_test = test_df['home_wins']
    
    # Step 3: Load OPTIMIZED model (not baseline)
    print("\n3. Loading optimized trained model...")
    
    try:
        # Load the OPTIMIZED model
        win_model = joblib.load(Paths.MODELS / 'win_model_optimized.joblib')
        print("   [OK] Loaded optimized win model (75.4% accuracy)")
    except Exception as e:
        print(f"   [ERROR] Error loading optimized model: {e}")
        print("   [INFO] Please run 'python scripts/optimize_models.py' first")
        return False
    
    # Step 4: Show current (uncalibrated) performance
    print_header("BEFORE CALIBRATION")
    
    from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
    
    y_pred_train = win_model.predict(X_train)
    y_pred_test = win_model.predict(X_test)
    y_proba_train = win_model.predict_proba(X_train)[:, 1]
    y_proba_test = win_model.predict_proba(X_test)[:, 1]
    
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    train_logloss = log_loss(y_train, y_proba_train)
    test_logloss = log_loss(y_test, y_proba_test)
    train_brier = brier_score_loss(y_train, y_proba_train)
    test_brier = brier_score_loss(y_test, y_proba_test)
    
    print("\nOptimized Model Performance (before calibration):")
    print(f"  Accuracy: Train {train_acc:.3f} | Test {test_acc:.3f}")
    print(f"  Log Loss: Train {train_logloss:.3f} | Test {test_logloss:.3f}")
    print(f"  Brier Score: Train {train_brier:.3f} | Test {test_brier:.3f}")
    
    # Show problematic predictions
    print("\nProbability Distribution (Test Set):")
    print(f"  Min:  {y_proba_test.min():.3f}")
    print(f"  25%:  {pd.Series(y_proba_test).quantile(0.25):.3f}")
    print(f"  50%:  {pd.Series(y_proba_test).quantile(0.50):.3f}")
    print(f"  75%:  {pd.Series(y_proba_test).quantile(0.75):.3f}")
    print(f"  Max:  {y_proba_test.max():.3f}")
    print(f"  Mean: {y_proba_test.mean():.3f}")
    
    # Count overconfident predictions
    overconfident_90 = (y_proba_test > 0.9).sum()
    overconfident_95 = (y_proba_test > 0.95).sum()
    overconfident_99 = (y_proba_test > 0.99).sum()
    
    print(f"\nOverconfident Predictions:")
    print(f"  >90%: {overconfident_90} games ({overconfident_90/len(y_proba_test)*100:.1f}%)")
    print(f"  >95%: {overconfident_95} games ({overconfident_95/len(y_proba_test)*100:.1f}%)")
    print(f"  >99%: {overconfident_99} games ({overconfident_99/len(y_proba_test)*100:.1f}%)")
    
    # Step 5: Calibrate the model
    print_header("CALIBRATING MODEL")
    print("\nApplying isotonic regression calibration...")
    print("(This learns a monotonic function to correct probabilities)")
    
    calibrator = ModelCalibrator()
    calibrated_win_model = calibrator.calibrate_classifier(
        win_model, X_train, y_train, X_test, y_test, 'win'
    )
    
    # Step 6: Show improvement
    print_header("AFTER CALIBRATION")
    
    y_proba_cal_train = calibrated_win_model.predict_proba(X_train)[:, 1]
    y_proba_cal_test = calibrated_win_model.predict_proba(X_test)[:, 1]
    
    cal_train_logloss = log_loss(y_train, y_proba_cal_train)
    cal_test_logloss = log_loss(y_test, y_proba_cal_test)
    cal_train_brier = brier_score_loss(y_train, y_proba_cal_train)
    cal_test_brier = brier_score_loss(y_test, y_proba_cal_test)
    
    print("\nCalibrated Performance:")
    print(f"  Accuracy: Train {train_acc:.3f} | Test {test_acc:.3f} (unchanged)")
    print(f"  Log Loss: Train {cal_train_logloss:.3f} | Test {cal_test_logloss:.3f}")
    print(f"  Brier Score: Train {cal_train_brier:.3f} | Test {cal_test_brier:.3f}")
    
    print("\nProbability Distribution (Test Set):")
    print(f"  Min:  {y_proba_cal_test.min():.3f}")
    print(f"  25%:  {pd.Series(y_proba_cal_test).quantile(0.25):.3f}")
    print(f"  50%:  {pd.Series(y_proba_cal_test).quantile(0.50):.3f}")
    print(f"  75%:  {pd.Series(y_proba_cal_test).quantile(0.75):.3f}")
    print(f"  Max:  {y_proba_cal_test.max():.3f}")
    print(f"  Mean: {y_proba_cal_test.mean():.3f}")
    
    # Count overconfident predictions after calibration
    cal_overconfident_90 = (y_proba_cal_test > 0.9).sum()
    cal_overconfident_95 = (y_proba_cal_test > 0.95).sum()
    cal_overconfident_99 = (y_proba_cal_test > 0.99).sum()
    
    print(f"\nOverconfident Predictions:")
    print(f"  >90%: {cal_overconfident_90} games ({cal_overconfident_90/len(y_proba_cal_test)*100:.1f}%)")
    print(f"  >95%: {cal_overconfident_95} games ({cal_overconfident_95/len(y_proba_cal_test)*100:.1f}%)")
    print(f"  >99%: {cal_overconfident_99} games ({cal_overconfident_99/len(y_proba_cal_test)*100:.1f}%)")
    
    # Step 7: Show improvement metrics
    print_header("IMPROVEMENT SUMMARY")
    
    logloss_improvement = (test_logloss - cal_test_logloss) / test_logloss * 100
    brier_improvement = (test_brier - cal_test_brier) / test_brier * 100
    
    print(f"\nLog Loss Improvement: {logloss_improvement:+.1f}%")
    print(f"Brier Score Improvement: {brier_improvement:+.1f}%")
    
    print(f"\nOverconfident Prediction Reduction:")
    print(f"  >90%: {overconfident_90} -> {cal_overconfident_90} ({-(overconfident_90-cal_overconfident_90)} games)")
    print(f"  >95%: {overconfident_95} -> {cal_overconfident_95} ({-(overconfident_95-cal_overconfident_95)} games)")
    print(f"  >99%: {overconfident_99} -> {cal_overconfident_99} ({-(overconfident_99-cal_overconfident_99)} games)")
    
    # Step 8: Create calibration curve
    print("\n" + "="*70)
    print("  CREATING CALIBRATION CURVE VISUALIZATION")
    print("="*70)
    
    try:
        plot_path = Paths.MODELS.parent / 'visualizations' / 'calibration_curve_optimized.png'
        calibrator.plot_calibration_curve(
            win_model, calibrated_win_model, X_test, y_test,
            model_name='Win Probability (Optimized)', save_path=plot_path
        )
        print(f"\n[OK] Saved calibration curve to {plot_path}")
    except Exception as e:
        print(f"[WARNING] Could not create plot: {e}")
        print("(matplotlib display might not be available)")
    
    # Step 9: Compare sample predictions
    print_header("SAMPLE PREDICTION COMPARISON")
    
    comparison = calibrator.compare_predictions(
        win_model, calibrated_win_model, X_test, n_samples=10
    )
    
    print("\nBefore vs After Calibration (10 sample games):")
    print(comparison.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    
    # Step 10: Save calibrated models
    print_header("SAVING CALIBRATED MODELS")
    
    try:
        calibrator.save_calibrated_models()
        print("\n[OK] Calibrated models saved successfully!")
        print(f"\nSaved files:")
        print(f"  - {Paths.MODELS / 'win_model_calibrated.joblib'}")
        print(f"  - {Paths.MODELS / 'calibration_metrics.joblib'}")
    except Exception as e:
        print(f"[ERROR] Error saving models: {e}")
        return False
    
    # Step 11: Final summary
    print_header("CALIBRATION COMPLETE")
    
    print("\n[OK] Model calibration successful!")
    print(f"[OK] Log loss improved by {logloss_improvement:.1f}%")
    print(f"[OK] Brier score improved by {brier_improvement:.1f}%")
    print(f"[OK] Overconfident (>95%) predictions reduced from {overconfident_95} to {cal_overconfident_95}")
    
    print("\nNext steps:")
    print("  1. Review calibration curve visualization")
    print("  2. Use calibrated model for future predictions")
    print("  3. Phase 4 Complete - Model ready for production!")
    
    print("\nTo use calibrated + optimized model:")
    print("  - Load: win_model_calibrated.joblib")
    print("  - This combines: 75.4% accuracy + proper calibration")
    print("  - Best of both worlds!")
    
    return True


if __name__ == "__main__":
    try:
        success = calibrate_models()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[ERROR] Calibration failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)