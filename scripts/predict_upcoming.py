#!/usr/bin/env python3
"""
Production Prediction Script
Generates predictions for upcoming NFL games using the optimized model
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import joblib
from config.settings import Paths
from src.data.loader import get_loader
from src.features.aggregator import TeamAggregator
from src.features.recent_form import RecentFormCalculator
from src.features.engineer import FeatureEngineer

def predict_week():
    """Generate predictions for upcoming games"""
    
    # Load optimized models
    win_model = joblib.load(Paths.MODELS / 'win_model_optimized.joblib')
    margin_model = joblib.load(Paths.MODELS / 'margin_model_optimized.joblib')
    total_model = joblib.load(Paths.MODELS / 'total_model_optimized.joblib')
    
    # Load current data
    loader = get_loader()
    schedules = loader.load_schedules()
    team_stats = loader.load_team_stats()
    
    # Engineer features
    aggregator = TeamAggregator()
    team_averages = aggregator.aggregate_team_stats(team_stats)
    
    form_calculator = RecentFormCalculator()
    recent_form = form_calculator.calculate_recent_form(schedules)
    
    engineer = FeatureEngineer()
    
    # Get upcoming games
    current_week = loader.get_current_week()
    upcoming = schedules[
        (schedules['season'] == 2025) &
        (schedules['week'] == current_week) &
        (schedules['away_score'].isna())
    ]
    
    # Create features
    features = engineer.create_game_features(upcoming, team_averages, recent_form)
    feature_cols = engineer.get_feature_importance_names(features)
    X = features[feature_cols]
    
    # Generate predictions
    win_proba = win_model.predict_proba(X)[:, 1]
    margins = margin_model.predict(X)
    totals = total_model.predict(X)
    
    # Display results
    for i, (_, game) in enumerate(features.iterrows()):
        away = game['away_team']
        home = game['home_team']
        home_win_prob = win_proba[i]
        
        print(f"\n{away} @ {home}")
        print(f"  Home win probability: {home_win_prob:.1%}")
        print(f"  Predicted margin: {margins[i]:+.1f} (positive = home favored)")
        print(f"  Predicted total: {totals[i]:.1f} points")

if __name__ == "__main__":
    predict_week()