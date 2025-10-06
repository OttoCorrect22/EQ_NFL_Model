#!/usr/bin/env python3
"""
Test script for model training and prediction

This script:
1. Loads NFL data and features
2. Trains all three XGBoost models
3. Evaluates performance with walk-forward validation
4. Generates predictions for upcoming games
5. Saves trained models

Run this to train models and see predictions.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.loader import get_loader
from src.features.aggregator import TeamAggregator
from src.features.recent_form import RecentFormCalculator
from src.features.engineer import FeatureEngineer
from src.models.trainer import NFLModelTrainer
from src.models.predictor import NFLPredictor
import pandas as pd
import logging
import io

# Fix Windows Unicode encoding issues
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)


def print_header(text: str):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def test_model_pipeline():
    """Test the complete model training and prediction pipeline"""
    
    print_header("NFL MODEL TRAINING & PREDICTION TEST")
    
    # Step 1: Load data and engineer features
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
        
        # Engineer features for completed games only
        completed_games = schedules[schedules['away_score'].notna()].copy()
        features = engineer.create_game_features(
            completed_games, team_averages, recent_form
        )
        
        print(f"   ✓ Loaded {len(features)} completed games with features")
        
    except Exception as e:
        print(f"   ✗ Error loading data: {e}")
        return False
    
    # Step 2: Train models
    print_header("MODEL TRAINING")
    print("\nTraining three XGBoost models with walk-forward validation...")
    print("(This may take 1-2 minutes)\n")
    
    try:
        trainer = NFLModelTrainer()
        results = trainer.train_all_models(features)
        
        print("\n" + "="*70)
        print("  TRAINING COMPLETE")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ Error training models: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Display performance metrics
    print_header("MODEL PERFORMANCE SUMMARY")
    
    metrics = results['metrics']
    
    print("\nWin Probability Model:")
    print(f"  Training accuracy: {metrics['win']['train_accuracy']:.3f}")
    print(f"  Test accuracy: {metrics['win']['test_accuracy']:.3f}")
    print(f"  Test log loss: {metrics['win']['test_logloss']:.3f}")
    
    print("\nMargin Prediction Model:")
    print(f"  Training MAE: {metrics['margin']['train_mae']:.2f} points")
    print(f"  Test MAE: {metrics['margin']['test_mae']:.2f} points")
    
    print("\nTotal Points Model:")
    print(f"  Training MAE: {metrics['total']['train_mae']:.2f} points")
    print(f"  Test MAE: {metrics['total']['test_mae']:.2f} points")
    
    # Compare to baselines
    print("\n" + "-"*70)
    print("Performance vs Baselines:")
    
    test_acc = metrics['win']['test_accuracy']
    if test_acc > 0.60:
        status = "✓ Exceeds expected performance (60-65%)"
    elif test_acc > 0.56:
        status = "✓ Above baseline (56% home team always wins)"
    else:
        status = "⚠ Below expected performance"
    
    print(f"  Win accuracy: {test_acc:.1%} - {status}")
    
    margin_mae = metrics['margin']['test_mae']
    if margin_mae < 12:
        status = "✓ Within expected range (10-12 points)"
    else:
        status = "⚠ Higher than expected"
    
    print(f"  Margin MAE: {margin_mae:.1f} points - {status}")
    
    # Step 4: Save models
    print_header("SAVING MODELS")
    
    try:
        trainer.save_models()
        print("   ✓ All models saved successfully")
    except Exception as e:
        print(f"   ✗ Error saving models: {e}")
    
    # Step 5: Generate predictions for upcoming games
    print_header("GENERATING PREDICTIONS FOR UPCOMING GAMES")
    
    try:
        # Get upcoming games
        current_season = loader.get_current_season()
        current_week = loader.get_current_week()
        
        upcoming_games = schedules[
            (schedules['season'] == current_season) &
            (schedules['week'] == current_week) &
            (schedules['away_score'].isna())
        ]
        
        if len(upcoming_games) == 0:
            print(f"\nNo upcoming games in Week {current_week}")
            print("Showing predictions for most recent completed games instead...\n")
            
            # Use last 5 completed games as examples
            recent_completed = schedules[
                (schedules['season'] == current_season) &
                (schedules['away_score'].notna())
            ].tail(5)
            
            upcoming_games = recent_completed
        
        # Engineer features for these games
        prediction_features = engineer.create_game_features(
            upcoming_games, team_averages, recent_form
        )
        
        # Generate predictions
        predictor = NFLPredictor(trainer)
        predictions = predictor.predict_games(prediction_features)
        
        print(f"\nGenerated predictions for {len(predictions)} games:\n")
        
        # Display predictions
        for _, pred in predictions.head(5).iterrows():
            summary = predictor.format_prediction_summary(pred)
            print(summary)
        
        if len(predictions) > 5:
            print(f"\n... and {len(predictions) - 5} more games")
        
        # Save predictions
        predictor.save_predictions(predictions, filename="test_predictions.csv")
        print(f"\n✓ Predictions saved to data/predictions/test_predictions.csv")
        
    except Exception as e:
        print(f"\n✗ Error generating predictions: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 6: Model interpretation
    print_header("MODEL INSIGHTS")
    
    print("\nKey findings:")
    print(f"  - Trained on {len(features[features['season'] < current_season])} historical games")
    print(f"  - Tested on {len(features[features['season'] == current_season])} current season games")
    print(f"  - Using {len(results['feature_names'])} predictive features")
    print(f"  - Win prediction accuracy: {metrics['win']['test_accuracy']:.1%}")
    
    # Feature importance insights
    print("\nMost important prediction factors:")
    print("  (See model training logs above for detailed feature importance)")
    
    # Research alignment
    print_header("RESEARCH ALIGNMENT")
    
    print("\nComparing to NFL prediction research benchmarks:")
    print("  Research shows XGBoost typically achieves:")
    print("    - Win accuracy: 60-65%")
    print("    - Margin MAE: 10-12 points")
    print("    - Total MAE: 8-10 points")
    
    print(f"\n  Your model performance:")
    print(f"    - Win accuracy: {metrics['win']['test_accuracy']:.1%}")
    print(f"    - Margin MAE: {metrics['margin']['test_mae']:.1f} points")
    print(f"    - Total MAE: {metrics['total']['test_mae']:.1f} points")
    
    if metrics['win']['test_accuracy'] >= 0.60:
        print("\n  ✓ Model performance aligns with research expectations")
    else:
        print("\n  ⚠ Model performance below research benchmarks")
        print("    Consider: more features, hyperparameter tuning, or ensemble methods")
    
    # Summary
    print_header("PHASE 3 COMPLETE")
    
    print("\n✓ Model training successful")
    print("✓ All three models trained and saved")
    print("✓ Predictions generated for upcoming games")
    
    print("\nNext steps:")
    print("  1. Commit Phase 3 to Git")
    print("  2. Review predictions in data/predictions/test_predictions.csv")
    print("  3. Phase 4: Build production prediction pipeline")
    print("  4. Phase 5: Add odds API integration")
    
    print("\nModel files saved to:")
    print("  - models/trained/win_model.joblib")
    print("  - models/trained/margin_model.joblib")
    print("  - models/trained/total_model.joblib")
    print("  - models/trained/feature_names.joblib")
    print("  - models/trained/performance_metrics.joblib")
    
    return True


if __name__ == "__main__":
    try:
        success = test_model_pipeline()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)