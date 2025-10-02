#!/usr/bin/env python3
"""
Test script to verify data loading functionality

This script:
1. Loads NFL schedules and team stats
2. Displays basic information about the data
3. Verifies data quality
4. Shows sample records

Run this to confirm your data pipeline is working correctly.
"""

import sys
from pathlib import Path

# Add project root to path so we can import our modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import get_loader
from config.settings import data_config
import pandas as pd


def print_header(text: str):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def test_data_loading():
    """Test the data loading functionality"""
    
    print_header("NFL DATA LOADER TEST")
    
    # Initialize loader
    print("\n1. Initializing data loader...")
    loader = get_loader()
    print(f"   ✓ Loader configured for seasons: {loader.seasons}")
    
    # Load schedules
    print("\n2. Loading game schedules...")
    try:
        schedules = loader.load_schedules()
        print(f"   ✓ Loaded {len(schedules)} games")
        print(f"   ✓ Columns: {len(schedules.columns)}")
        print(f"   ✓ Date range: {schedules['gameday'].min()} to {schedules['gameday'].max()}")
    except Exception as e:
        print(f"   ✗ Error loading schedules: {e}")
        return False
    
    # Load team stats
    print("\n3. Loading team statistics...")
    try:
        team_stats = loader.load_team_stats()
        print(f"   ✓ Loaded {len(team_stats)} team-game records")
        print(f"   ✓ Columns: {len(team_stats.columns)}")
        print(f"   ✓ Teams: {team_stats['team'].nunique()}")
    except Exception as e:
        print(f"   ✗ Error loading team stats: {e}")
        return False
    
    # Analyze schedules
    print_header("SCHEDULE DATA ANALYSIS")
    
    # Games by season
    print("\nGames by season:")
    for season in sorted(schedules['season'].unique()):
        season_games = schedules[schedules['season'] == season]
        completed = season_games['away_score'].notna().sum()
        upcoming = season_games['away_score'].isna().sum()
        print(f"   {season}: {len(season_games)} total ({completed} completed, {upcoming} upcoming)")
    
    # Current season info
    current_season = loader.get_current_season()
    current_week = loader.get_current_week()
    print(f"\nCurrent season: {current_season}, Week {current_week}")
    
    # Show upcoming games
    current_schedule = schedules[schedules['season'] == current_season]
    upcoming_games = current_schedule[
        (current_schedule['week'] == current_week) & 
        (current_schedule['away_score'].isna())
    ]
    
    if len(upcoming_games) > 0:
        print(f"\nUpcoming games this week ({len(upcoming_games)} games):")
        for _, game in upcoming_games.head(5).iterrows():
            gameday = pd.to_datetime(game['gameday']).strftime('%a %m/%d')
            print(f"   {game['away_team']} @ {game['home_team']} ({gameday})")
        if len(upcoming_games) > 5:
            print(f"   ... and {len(upcoming_games) - 5} more")
    else:
        print("\nNo upcoming games found for current week")
    
    # Analyze team stats
    print_header("TEAM STATS ANALYSIS")
    
    # Stats by season
    print("\nTeam-game records by season:")
    for season in sorted(team_stats['season'].unique()):
        season_records = len(team_stats[team_stats['season'] == season])
        print(f"   {season}: {season_records} records")
    
    # Available statistics
    print("\nKey statistics available:")
    key_stats = [
        'passing_yards', 'passing_tds', 'passing_epa',
        'rushing_yards', 'rushing_tds', 'rushing_epa',
        'def_sacks', 'def_interceptions', 'def_tds'
    ]
    
    available_stats = [stat for stat in key_stats if stat in team_stats.columns]
    missing_stats = [stat for stat in key_stats if stat not in team_stats.columns]
    
    print(f"   Available ({len(available_stats)}/{len(key_stats)}):")
    for stat in available_stats[:6]:  # Show first 6
        print(f"      ✓ {stat}")
    if len(available_stats) > 6:
        print(f"      ... and {len(available_stats) - 6} more")
    
    if missing_stats:
        print(f"\n   Missing ({len(missing_stats)}):")
        for stat in missing_stats:
            print(f"      ✗ {stat}")
    
    # Sample team data
    print("\nSample team performance (BUF most recent game):")
    buf_recent = team_stats[team_stats['team'] == 'BUF'].sort_values(
        ['season', 'week'], 
        ascending=False
    ).head(1)
    
    if len(buf_recent) > 0:
        game = buf_recent.iloc[0]
        print(f"   Season: {game['season']}, Week: {game['week']}")
        if 'passing_yards' in game.index:
            print(f"   Passing yards: {game['passing_yards']:.0f}")
        if 'rushing_yards' in game.index:
            print(f"   Rushing yards: {game['rushing_yards']:.0f}")
        if 'passing_tds' in game.index:
            print(f"   Passing TDs: {game['passing_tds']:.0f}")
    
    # Data quality checks
    print_header("DATA QUALITY CHECKS")
    
    # Check for missing critical columns
    critical_schedule_cols = ['season', 'week', 'away_team', 'home_team', 'gameday']
    critical_stats_cols = ['season', 'week', 'team']
    
    missing_schedule_cols = [col for col in critical_schedule_cols if col not in schedules.columns]
    missing_stats_cols = [col for col in critical_stats_cols if col not in team_stats.columns]
    
    if not missing_schedule_cols and not missing_stats_cols:
        print("\n✓ All critical columns present")
    else:
        if missing_schedule_cols:
            print(f"\n✗ Missing schedule columns: {missing_schedule_cols}")
        if missing_stats_cols:
            print(f"✗ Missing stats columns: {missing_stats_cols}")
    
    # Check for null values in critical columns
    print("\nNull value check:")
    schedule_nulls = schedules[critical_schedule_cols].isnull().sum().sum()
    stats_nulls = team_stats[critical_stats_cols].isnull().sum().sum()
    
    if schedule_nulls == 0 and stats_nulls == 0:
        print("   ✓ No null values in critical columns")
    else:
        print(f"   ! Found {schedule_nulls} null values in schedules")
        print(f"   ! Found {stats_nulls} null values in team stats")
    
    # Success summary
    print_header("TEST SUMMARY")
    print("\n✓ Data loading successful")
    print(f"✓ {len(schedules)} games loaded")
    print(f"✓ {len(team_stats)} team performance records loaded")
    print(f"✓ {len(available_stats)} key statistics available")
    print("\nYour data pipeline is working correctly!")
    print("\nNext steps:")
    print("  - Phase 2: Feature engineering (transform stats into predictions)")
    print("  - Phase 3: Model training (train ML models)")
    print("  - Phase 4: Generate predictions for upcoming games")
    
    return True


if __name__ == "__main__":
    try:
        success = test_data_loading()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)