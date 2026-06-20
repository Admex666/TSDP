# World Cup Match Enjoyability Predictor

This project implements a machine learning pipeline to predict the "enjoyability" of FIFA World Cup matches (specifically from the 2022 World Cup in Qatar) using statistical match indicators from SofaScore. The target enjoyability metric is the **IMDb user rating** for each match episode.

---

## Project Structure

*   **`sofascore_data.json`**: Aggregated raw JSON containing details, statistics, and incidents for all 64 matches of the 2022 World Cup (fetched via browser console).
*   **`all_imdb_episodes_deduped.csv`**: IMDb ratings and metadata for all 64 matches.
*   **`prepare_data.py`**: Merges the IMDb and SofaScore databases, resolves team name and stage mappings, engineers excitement features, and saves the final dataset to `WC_matches_final.csv`.
*   **`train_model.py`**: Performs exploratory correlation analysis, scales features, runs 5-Fold Cross-Validation on multiple regression models, trains the best model, evaluates feature coefficients/importances, and checks predictions.
*   **`WC_matches_final.csv`**: The clean, final tabular dataset containing engineered features, actual IMDb ratings, and model predictions.

---

## Feature Engineering

From the raw SofaScore details, stats, and incidents, we engineered the following variables:

1.  **Goals & Drama**
    *   `total_goals`: Sum of goals in normal + extra time.
    *   `goal_difference`: Absolute margin of victory.
    *   `lead_changes`: Number of times the lead changed hands (e.g. A leads -> draw -> B leads).
    *   `comeback_win_draw`: 1 if a team trailing at any point managed to draw or win, 0 otherwise.
    *   `late_goals`: Number of goals scored at/after the 80th minute or in extra time.
    *   `time_of_last_goal`: The minute the final goal was scored (normal + extra time).
    *   `extra_time` & `penalty_shootout`: Structural indicators for overtime and shootouts.
2.  **Offensive Quality**
    *   `total_shots` & `total_shots_on_target`: Volume and precision of attacks.
    *   `total_xg`: Combined Expected Goals (xG) of both teams.
    *   `xg_difference`: Match balance indicator (dominant vs. evenly contested).
    *   `big_chances_total` & `big_chances_missed`: Number of high-quality chances created and wasted.
3.  **Intensity & Incidents**
    *   `possession_imbalance`: Domination from 50% (e.g. |65% - 50%| = 15%).
    *   `fouls_total` & `corner_kicks`: Traditional flow metrics.
    *   `red_cards` & `yellow_cards`: Match disciplinary drama.
    *   `penalties_awarded`: Total penalties taken in normal + extra time (scored + missed).
4.  **Stakes Context**
    *   `stage_stakes`: Ordinal mapping of the tournament stage (Group stage: 1.0, Round of 16: 2.0, Quarterfinals: 3.0, 3rd place: 3.5, Semifinals: 4.0, Final: 5.0).

---

## Key Findings: What Makes a Match Enjoyable?

Here is the linear correlation of each numeric feature with the actual IMDb rating:

| Feature | Correlation with IMDb Rating | Interpretation |
| :--- | :---: | :--- |
| **`time_of_last_goal`** | **0.460** | Matches with very late goals (e.g., 90'+ or ET) are much more exciting. |
| **`comeback_win_draw`** | **0.444** | Matches featuring comebacks have higher enjoyability ratings. |
| **`total_goals`** | **0.439** | High-scoring matches are naturally rated higher. |
| **`total_xg`** | **0.417** | Match enjoyment is heavily tied to the quality of goalscoring chances. |
| **`total_shots_on_target`**| **0.390** | More shots on target keep the audience engaged. |
| **`penalties_awarded`** | **0.372** | Awarded penalties create high-tension moments. |
| **`penalty_shootout`** | **0.359** | Penalty shootouts represent the absolute peak of drama. |
| **`extra_time`** | **0.359** | Overtime matches are highly rated. |
| **`red_cards`** | **0.304** | Red cards add narrative controversy and tactical shifts. |
| **`lead_changes`** | **0.285** | Volatile games where leadership swings back and forth are more fun. |
| **`goal_difference`** | **-0.039** | Surprisingly near 0, because blowouts (e.g. Portugal 6-1, Spain 7-0) can still be rated highly due to goal volume, while boring close matches drag down the correlation. |

---

## Model Evaluation (5-Fold Cross-Validation)

Given the small dataset size (64 rows), we tested simple linear models alongside regularized and tree-based regressors to prevent overfitting:

| Model | CV MAE | CV RMSE | CV $R^2$ |
| :--- | :---: | :---: | :---: |
| **Linear Regression** | **0.558** | **0.698** | **0.314** |
| Ridge Regression | 0.593 | 0.719 | 0.268 |
| XGBoost (depth=2) | 0.649 | 0.791 | 0.127 |
| Random Forest (depth=3) | 0.656 | 0.820 | 0.069 |
| Random Forest (depth=2) | 0.658 | 0.833 | 0.034 |

**Linear Regression** performed the best, explaining **31.4%** of the variance in IMDb ratings on unseen folds, with an average absolute prediction error of **0.56 points** on a 10-point scale.

### Best Model Coefficient Insights (Linear Regression)
*   **`time_of_last_goal` (+0.577)**: The single strongest driver. Drama in the dying minutes increases ratings.
*   **`total_xg` (+0.473)**: High attacking quality yields higher enjoyment.
*   **`goal_difference` (-0.451)**: In a multivariate setting, when holding other variables constant, **closer games are strongly preferred**.
*   **`lead_changes` (+0.432)**: Lead changes add huge value.
*   **`red_cards` (+0.314)**: Sending offs boost drama and ratings.
*   **`stage_stakes` (+0.218)**: Knockout and final rounds get a baseline rating boost due to high stakes.

---

## Predictions for Famous Matches

Here are the predictions made by our trained Linear Regression model:

*   **Argentina vs France (Final)**: IMDb Rating = **9.7**, Predicted = **9.56** (Error: +0.14)
    *   *Why?* 6 goals, a late penalty equalizer in ET, penalty shootout, massive xG, high stakes.
*   **Cameroon vs Serbia (3-3 Group Stage)**: IMDb Rating = **8.3**, Predicted = **7.84** (Error: +0.46)
    *   *Why?* 6 goals, lead change, late goals, major comeback.
*   **Croatia vs Brazil (Quarter-Finals)**: IMDb Rating = **7.5**, Predicted = **7.77** (Error: -0.27)
    *   *Why?* Extra time, late equalizer, shootout, high stakes.
*   **Qatar vs Ecuador (Opening Match)**: IMDb Rating = **5.8**, Predicted = **6.20** (Error: -0.40)
    *   *Why?* Only 2 goals, early finish, low shots on target, one-sided.

---

## How to Run

1.  Ensure you have your requirements installed:
    ```bash
    pip install pandas numpy scikit-learn xgboost
    ```
2.  Run the data preparation script:
    ```bash
    python prepare_data.py
    ```
3.  Run the model training and evaluation script:
    ```bash
    python train_model.py
    ```
