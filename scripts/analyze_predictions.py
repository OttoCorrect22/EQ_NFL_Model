# scripts/analyze_predictions.py
import pandas as pd
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import get_loader
import io

# Fix Windows Unicode encoding issues
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load predictions
predictions = pd.read_csv(project_root / 'data/predictions/test_predictions.csv')

# Load actual results for comparison
loader = get_loader()
schedules = loader.load_schedules()

# Filter to 2025 completed games
actual_2025 = schedules[
    (schedules['season'] == 2025) & 
    (schedules['away_score'].notna())
].copy()

print("="*70)
print("PREDICTION ANALYSIS")
print("="*70)

# Show confidence distribution
print("\nConfidence Distribution:")
print(predictions['confidence'].describe())

# Show most confident predictions
print("\nTop 5 Most Confident Predictions:")
print(predictions.nlargest(5, 'confidence')[
    ['away_team', 'home_team', 'predicted_winner', 'confidence']
])

# Show least confident (closest games)
print("\nTop 5 Closest Games (Least Confident):")
print(predictions.nsmallest(5, 'confidence')[
    ['away_team', 'home_team', 'predicted_winner', 'confidence']
])

# Predicted margin distribution
print("\nPredicted Margin Distribution:")
print(predictions['predicted_margin'].describe())

# Total points distribution
print("\nPredicted Total Points Distribution:")
print(predictions['predicted_total'].describe())