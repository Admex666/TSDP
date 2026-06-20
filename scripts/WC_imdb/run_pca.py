import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import os
import sys

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    data_path = r"e:\Data\TSDP\scripts\WC_imdb\WC_matches_final.csv"
    if not os.path.exists(data_path):
        print(f"Error: Dataset {data_path} not found. Please run prepare_data.py first.")
        return

    # Load dataset
    df = pd.read_csv(data_path)
    
    stage_stakes_map = {
        "Group stage": 1.0,
        "Round of 16": 2.0,
        "Quarterfinals": 3.0,
        "Match for 3rd place": 3.5,
        "Semifinals": 4.0,
        "Final": 5.0
    }
    df["stage_stakes"] = df["stage"].map(stage_stakes_map)

    feature_cols = [
        "total_goals", "goal_difference", "lead_changes", "comeback_win_draw", 
        "late_goals", "time_of_last_goal", "extra_time", "penalty_shootout", 
        "total_shots", "total_shots_on_target", "total_xg", "xg_difference", 
        "big_chances_total", "big_chances_missed", "possession_imbalance", 
        "fouls_total", "corner_kicks", "red_cards", "yellow_cards", 
        "penalties_awarded", "stage_stakes"
    ]
    
    X = df[feature_cols]
    y = df["rating"]

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    n_components = 5
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # 1. Explained Variance
    print("=== PCA Explained Variance ===")
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    for i in range(n_components):
        print(f"  - Component {i+1}: {explained_var[i]*100:5.2f}% (Cumulative: {cumulative_var[i]*100:5.2f}%)")

    # 2. Analyze Component Loadings
    print("\n=== Interpretation of Top 3 Components ===")
    loadings = pd.DataFrame(pca.components_.T, columns=[f"PC{i+1}" for i in range(n_components)], index=feature_cols)
    
    for i in range(3):
        print(f"\n--- Component {i+1} Top Drivers (Highest Loading Absolute Values) ---")
        pc_col = f"PC{i+1}"
        sorted_loadings = loadings[pc_col].sort_values(ascending=False)
        print("  Positive drivers:")
        for idx, val in sorted_loadings.head(4).items():
            print(f"    * {idx:25s}: {val:+.3f}")
        print("  Negative drivers:")
        for idx, val in sorted_loadings.tail(4).items():
            print(f"    * {idx:25s}: {val:+.3f}")

    # 3. Principal Component Regression (PCR) vs Standard Linear Regression (5-Fold CV)
    print("\n=== Principal Component Regression (PCR) 5-Fold CV ===")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Standard Linear Regression baseline from train_model
    baseline_mae = 0.558
    
    for k in range(1, n_components + 1):
        cv_maes = []
        cv_r2s = []
        for train_idx, val_idx in kf.split(X, y):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Apply PCA fit on training folds only
            pca_fold = PCA(n_components=k)
            X_train_pca = pca_fold.fit_transform(X_train)
            X_val_pca = pca_fold.transform(X_val)
            
            model = LinearRegression()
            model.fit(X_train_pca, y_train)
            preds = model.predict(X_val_pca)
            
            cv_maes.append(mean_absolute_error(y_val, preds))
            cv_r2s.append(r2_score(y_val, preds))
            
        print(f"  - PCR with {k} Components: CV MAE = {np.mean(cv_maes):.3f}, CV R² = {np.mean(cv_r2s):.3f}")

    # 4. Coordinates of famous matches in PCA 2D space
    print("\n=== Famous Match Coordinates in PCA Space (PC1 vs PC2) ===")
    famous_matches = [
        ("Argentina", "France", "Final"),
        ("Cameroon", "Serbia", "Group 3-3"),
        ("Qatar", "Ecuador", "Opening 0-2"),
        ("Croatia", "Brazil", "QF ET/Pens"),
        ("Morocco", "Spain", "R16 ET/Pens")
    ]
    
    for h, a, note in famous_matches:
        match_idx = df[((df["home_team"] == h) & (df["away_team"] == a)) | ((df["home_team"] == a) & (df["away_team"] == h))].index
        if len(match_idx) > 0:
            idx = match_idx[0]
            print(f"  - {h} vs {a} ({note}): PC1 (Drama) = {X_pca[idx, 0]:+.2f}, PC2 (Offense) = {X_pca[idx, 1]:+.2f} | Rating = {df.loc[idx, 'rating']:.1f}")

    # 5. Plotting (Save visualization image)
    if HAS_PLOT:
        plt.figure(figsize=(10, 8))
        # Scatter plot colored by rating
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="coolwarm", s=100, edgecolors='black', alpha=0.8)
        cbar = plt.colorbar(scatter)
        cbar.set_label('IMDb Enjoyability Rating', rotation=270, labelpad=15)
        
        # Add labels for famous matches
        for h, a, note in famous_matches:
            match_idx = df[((df["home_team"] == h) & (df["away_team"] == a)) | ((df["home_team"] == a) & (df["away_team"] == h))].index
            if len(match_idx) > 0:
                idx = match_idx[0]
                plt.annotate(f"{h}-{a}", (X_pca[idx, 0] + 0.1, X_pca[idx, 1] + 0.1), fontsize=9, fontweight='bold')

        plt.title("World Cup 2022 Matches in 2D PCA Space", fontsize=14, fontweight='bold')
        plt.xlabel("PC1 (Match Intensity & Drama Loading)", fontsize=12)
        plt.ylabel("PC2 (Offensive Production & xG Loading)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plot_path = r"e:\Data\TSDP\scripts\WC_imdb\pca_visualization.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nSUCCESS! 2D PCA scatter plot saved to: {plot_path}")
    else:
        print("\nNote: matplotlib/seaborn is not installed. Skipping plot generation.")

if __name__ == "__main__":
    main()
