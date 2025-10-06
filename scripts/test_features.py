#!/usr/bin/env python3
"""
Test script for feature engineering pipeline

This script:
1. Loads NFL data
2. Aggregates team statistics
3. Calculates recent form
4. Engineers game features
5. Shows sample features for real matchups

Run this to verify the feature engineering pipeline works correctly.
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
import pandas as pd
import io

# Fix Windows Unicode encoding issues
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def print_header(text: str):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def test_feature_pipeline():
    """Test the complete feature engineering pipeline"""
    
    print_header("FEATURE ENGINEERING PIPELINE TEST")
    
    # Step 1: Load data
    print("\n1. Loading NFL data...")
    loader = get_loader()
    
    try:
        schedules = loader.load_schedules()
        team_stats = loader.load_team_stats()
        print(f"   ✓ Loaded {len(schedules)} games")
        print(f"   ✓ Loaded {len(team_stats)} team-game records")
    except Exception as e:
        print(f"   ✗ Error loading data: {e}")
        return False
    
    # Step 2: Aggregate team statistics
    print("\n2. Aggregating team statistics...")
    try:
        aggregator = TeamAggregator()
        team_averages = aggregator.aggregate_team_stats(team_stats)
        print(f"   ✓ Aggregated stats for {len(team_averages)} teams")
        print(f"   ✓ Features per team: {len(team_averages.columns)}")
    except Exception as e:
        print(f"   ✗ Error aggregating stats: {e}")
        return False
    
    # Step 3: Calculate recent form
    print("\n3. Calculating recent form...")
    try:
        form_calculator = RecentFormCalculator()
        recent_form = form_calculator.calculate_recent_form(schedules)
        print(f"   ✓ Calculated form for {len(recent_form)} teams")
        print(f"   ✓ Form metrics per team: {len(recent_form.columns)}")
    except Exception as e:
        print(f"   ✗ Error calculating form: {e}")
        return False
    
    # Step 4: Engineer game features
    print("\n4. Engineering game features...")
    try:
        engineer = FeatureEngineer()
        
        # Engineer features for completed games only (for testing)
        completed_games = schedules[schedules['away_score'].notna()].copy()
        features = engineer.create_game_features(
            completed_games, team_averages, recent_form
        )
        
        print(f"   ✓ Created features for {len(features)} games")
        print(f"   ✓ Features per game: {len(features.columns)}")
        
        # Get feature list for model training
        feature_cols = engineer.get_feature_importance_names(features)
        print(f"   ✓ Predictive features: {len(feature_cols)}")
        
    except Exception as e:
        print(f"   ✗ Error engineering features: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Analysis: Show sample team profiles
    print_header("TEAM PROFILE ANALYSIS")
    
    sample_teams = ['BUF', 'KC', 'SF', 'DET']
    available_teams = [t for t in sample_teams if t in team_averages.index]
    
    if available_teams:
        sample_team = available_teams[0]
        print(f"\nSample team profile: {sample_team}")
        print("-" * 40)
        
        team_profile = team_averages.loc[sample_team]
        
        # Show key stats
        key_stats = {
            'total_offense': 'Total Offense',
            'total_epa': 'Total EPA',
            'offensive_efficiency': 'Offensive Efficiency',
            'defensive_efficiency': 'Defensive Efficiency',
            'turnover_differential': 'Turnover Differential'
        }
        
        for stat, label in key_stats.items():
            if stat in team_profile.index:
                print(f"  {label}: {team_profile[stat]:.2f}")
    
    # Analysis: Show recent form
    print_header("RECENT FORM ANALYSIS")
    
    if available_teams:
        sample_team = available_teams[0]
        print(f"\n{sample_team} - Last 4 Games:")
        print("-" * 40)
        
        form = recent_form.loc[sample_team]
        
        print(f"  Record: {int(form['wins'])}-{int(form['losses'])}")
        print(f"  Win %: {form['win_pct']:.1%}")
        print(f"  Avg Points Scored: {form['avg_points_scored']:.1f}")
        print(f"  Avg Points Allowed: {form['avg_points_allowed']:.1f}")
        print(f"  Avg Margin: {form['avg_margin']:+.1f}")
        print(f"  Current Streak: {int(form['current_streak'])} {'W' if form['current_streak'] > 0 else 'L' if form['current_streak'] < 0 else '-'}")
        print(f"  Scoring Trend: {form['scoring_trend']:+.1f} (per game)")
    
    # Analysis: Show sample matchup features
    print_header("SAMPLE MATCHUP FEATURES")
    
    # Get a recent completed game
    recent_game = features.iloc[-1]
    away = recent_game['away_team']
    home = recent_game['home_team']
    
    print(f"\nMATCHUP: {away} @ {home}")
    print(f"Season {int(recent_game['season'])}, Week {int(recent_game['week'])}")
    print("-" * 40)
    
    # Show key differential features
    print("\nKey Differentials (positive favors home team):")
    diff_features = {
        'diff_total_offense': 'Total Offense',
        'diff_total_epa': 'EPA',
        'diff_offensive_efficiency': 'Off. Efficiency',
        'diff_defensive_efficiency': 'Def. Efficiency',
        'diff_recent_wins': 'Recent Wins',
        'diff_recent_avg_margin': 'Recent Margin'
    }
    
    for feat, label in diff_features.items():
        if feat in recent_game.index:
            value = recent_game[feat]
            favors = home if value > 0 else away if value < 0 else "Even"
            print(f"  {label}: {value:+.2f} (favors {favors})")
    
    # Show actual result if available
    if 'home_score' in recent_game.index:
        home_score = int(recent_game['home_score'])
        away_score = int(recent_game['away_score'])
        winner = home if home_score > away_score else away
        margin = abs(home_score - away_score)
        
        print(f"\nActual Result:")
        print(f"  {away} {away_score} - {home_score} {home}")
        print(f"  Winner: {winner} by {margin} points")
    
    # Feature statistics
    print_header("FEATURE STATISTICS")
    
    # Show distribution of key features
    print("\nFeature ranges (for model training):")
    
    key_model_features = [
        'diff_total_epa',
        'diff_offensive_efficiency',
        'diff_recent_wins',
        'home_advantage'
    ]
    
    for feat in key_model_features:
        if feat in features.columns:
            values = features[feat]
            print(f"\n  {feat}:")
            print(f"    Min: {values.min():.2f}")
            print(f"    Mean: {values.mean():.2f}")
            print(f"    Max: {values.max():.2f}")
            print(f"    Std Dev: {values.std():.2f}")
    
    # Data quality check
    print_header("DATA QUALITY CHECK")
    
    # Check for missing values in key features
    print("\nMissing value check:")
    missing_counts = features[feature_cols].isnull().sum()
    total_missing = missing_counts.sum()
    
    if total_missing == 0:
        print("   ✓ No missing values in feature columns")
    else:
        print(f"   ! Found {total_missing} missing values")
        print("\n   Columns with missing values:")
        for col in missing_counts[missing_counts > 0].index:
            print(f"      {col}: {missing_counts[col]}")
    
    # Check target variable distribution
    if 'home_wins' in features.columns:
        home_wins = features['home_wins'].sum()
        away_wins = len(features) - home_wins
        home_pct = home_wins / len(features)
        
        print(f"\nTarget variable distribution:")
        print(f"   Home wins: {home_wins} ({home_pct:.1%})")
        print(f"   Away wins: {away_wins} ({(1-home_pct):.1%})")
        
        # Check for reasonable home field advantage
        if 0.50 < home_pct < 0.60:
            print("   ✓ Home field advantage looks reasonable")
        else:
            print(f"   ! Home field advantage seems unusual")
    
    # Summary
    print_header("TEST SUMMARY")
    
    print("\n✓ Feature engineering pipeline working correctly")
    print(f"\nPipeline output:")
    print(f"  - {len(team_averages)} team profiles created")
    print(f"  - {len(recent_form)} team form profiles created")
    print(f"  - {len(features)} games with features")
    print(f"  - {len(feature_cols)} predictive features per game")
    
    print("\nFeature categories:")
    print(f"  - Team averages: {len([c for c in feature_cols if 'away_' in c or 'home_' in c])}")
    print(f"  - Differentials: {len([c for c in feature_cols if 'diff_' in c])}")
    print(f"  - Recent form: {len([c for c in feature_cols if 'recent_' in c])}")
    print(f"  - Context: {len([c for c in feature_cols if c in ['home_advantage', 'rest_advantage', 'spread_line', 'total_line']])}")
    
    print("\n✓ Phase 2 complete - Ready for model training!")
    
    print("\nNext steps:")
    print("  - Commit Phase 2 to Git")
    print("  - Phase 3: Train prediction models")
    print("  - Phase 4: Generate predictions for upcoming games")
    
    return True


if __name__ == "__main__":
    try:
        success = test_feature_pipeline()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)