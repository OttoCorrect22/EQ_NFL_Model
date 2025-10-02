# EQ NFL Prediction Model

Machine learning system for predicting NFL game outcomes using historical data and advanced statistics.

## Project Status

**Phase 1: Complete** - Data collection and pipeline working
- Data loader implemented
- Configuration system in place
- Quality verification passed

**Phase 2: Complete** - Feature engineering pipeline
- Team statistics aggregation with season weighting
- Recent form calculation (last 4 games)
- Game-level feature engineering
- 115 predictive features per game

**Next: Phase 3** - Model training

## Quick Start

### Installation

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### Test Data Pipeline

```bash
python scripts/test_data_load.py
```

This verifies the data pipeline is working and shows current NFL data availability.

## Project Structure

```
EQ_NFL_Model/
├── config/
│   └── settings.py          # Configuration management
├── src/
│   ├── data/
│   │   └── loader.py         # NFL data loading from nflreadpy
│   └── features/
│       ├── aggregator.py     # Team statistics aggregation
│       ├── recent_form.py    # Recent form calculation
│       └── engineer.py       # Game feature engineering
├── scripts/
│   ├── test_data_load.py    # Data pipeline verification
│   └── test_features.py     # Feature pipeline verification
├── data/
│   ├── raw/                 # Raw NFL data and cache
│   ├── processed/           # Processed features
│   └── predictions/         # Model outputs
└── models/
    └── trained/             # Saved ML models
```

## Data Source

Uses [nflreadpy](https://github.com/nflverse/nflreadpy) for NFL data:
- Game schedules and results (2023-2025)
- Team statistics (passing, rushing, defense)
- Advanced metrics (EPA, efficiency)
- Betting lines and context data

## Configuration

Edit `config/settings.py` to customize:
- Seasons to include in training
- Cache settings
- Model parameters
- Feature engineering options

## Current Capabilities

**Data Loading:**
- Schedules: 842 games across 3 seasons
- Team stats: 1268 team-game records
- 102 statistical columns available
- Automatic caching for performance

**Feature Engineering:**
- Team aggregation with season weighting (2025: 3x, 2024: 2x, 2023: 1x)
- Recent form analysis (last 4 games per team)
- 115 predictive features per game including:
  - Team averages (passing, rushing, defense, EPA)
  - Differentials (home vs away matchups)
  - Recent form metrics (momentum, trends)
  - Context features (home advantage, rest, betting lines)

**Data Quality:**
- All critical columns validated
- No missing values in feature columns
- 56.2% home win rate (realistic home field advantage)
- Ready for model training

## Development Phases

### Phase 1: Data Foundation ✓
- [x] Project structure
- [x] Configuration system
- [x] Data loader with caching
- [x] Quality verification
- [x] Documentation

### Phase 2: Feature Engineering ✓
- [x] Team statistics aggregation
- [x] Recent form calculation
- [x] Feature engineering pipeline
- [x] Feature documentation
- [x] Testing and verification

### Phase 3: Model Training
- [ ] Win probability model
- [ ] Score prediction models
- [ ] Model validation
- [ ] Performance metrics

### Phase 4: Prediction Pipeline
- [ ] Prediction generation
- [ ] Output formatting
- [ ] Confidence calibration

### Phase 5: Production Features
- [ ] Odds API integration
- [ ] Dashboard interface
- [ ] Automated updates

## Technical Details

**Data Pipeline:**
1. `NFLDataLoader` fetches data from nflreadpy
2. Converts Polars to Pandas for compatibility
3. Caches data locally (24hr default)
4. Validates data quality

**Configuration System:**
- `DataConfig`: Data sources and storage
- `ModelConfig`: ML parameters
- `PredictionConfig`: Output settings
- `FeatureConfig`: Feature engineering options

## Version Control

Initialize Git repository:
```bash
git init
git add .
git commit -m "Phase 1: Data pipeline complete"
```

## Testing

Verify the data pipeline:
```bash
python scripts/test_data_load.py
```

Verify the feature engineering pipeline:
```bash
python scripts/test_features.py
```

Expected output (test_features.py):
- 634 games with features
- 115 predictive features per game
- All quality checks passed
- Sample matchup analysis displayed

## Dependencies

Core requirements:
- Python 3.13.7
- pandas >= 2.0.0
- nflreadpy >= 0.1.0
- scikit-learn >= 1.3.0

See `requirements.txt` for complete list.

## Notes

**Why nflreadpy:**
- Comprehensive NFL data coverage
- Maintained by nflverse community
- Built-in caching
- Regular updates during season

**Data Weighting:**
Current season games weighted 3x more than historical games for recency.

**Performance:**
First data load takes 30-60 seconds (downloads from source). Subsequent loads use cache and complete in <1 second.

## Next Steps

1. Commit Phase 1 to Git
2. Begin Phase 2: Feature Engineering
3. Build team statistics aggregation
4. Calculate recent form metrics
5. Create feature engineering pipeline

---

**Project:** EQ NFL Model  
**Version:** Phase 2 Complete  
**Last Updated:** October 2, 2025