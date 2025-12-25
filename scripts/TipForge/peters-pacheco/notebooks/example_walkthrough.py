# %% [markdown]
# # Football Match Prediction Walkthrough
# This notebook demonstrates the end-to-end usage of the prediction pipeline.

# %%
import pandas as pd
import numpy as np
from src.scraping.fbref_loader import FBrefDataLoader
from src.features.builder import LineupFeatureBuilder
from src.models.regression import GoalRegressionModel
from src.betting.backtest import Backtester

# %% [markdown]
# ## 1. Data Loading
# Initialize the loader and fetch schedule + stats.

# %%
loader = FBrefDataLoader(data_dir="../data")
# schedule = loader.load_match_schedule("2023-2024", "9")
# stats = loader.load_player_season_stats("2023-2024", "9")
print("Data loader initialized.")

# %% [markdown]
# ## 2. Feature Engineering
# Build rolling lineup features.

# %%
builder = LineupFeatureBuilder(loader)
# features = []
# for _, match in schedule.iterrows():
#     # fetch lineups...
#     # feat = builder.build_features_for_match(...)
#     # features.append(feat)
# features_df = pd.DataFrame(features)
print("Feature builder initialized.")

# %% [markdown]
# ## 3. Modeling
# Train SVR models for Goal Prediction.

# %%
model = GoalRegressionModel()
# model.train(X_train, y_train)
print("Model initialized.")

# %% [markdown]
# ## 4. Backtesting
# Run chronological backtest and evaluate ROI.

# %%
# backtester = Backtester(features_df, schedule)
# backtester.run(start_date="2024-01-01")
# results = backtester.get_results_df()
# print(f"Total ROI: {results['Result'].sum() / results['Stake'].sum():.2%}")
print("Backtester ready.")
