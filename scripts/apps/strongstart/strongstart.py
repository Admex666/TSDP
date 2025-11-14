#%% imports
import os
import sys
import requests
from datetime import datetime

# Mentési mappa
# Elfogadott working directory-k
allowed_dirs = [
    r"C:\Users\Adam\..Data",
    r"C:\Users\Adam\.Data files"
]
# Ellenőrizzük, melyik elérhető
found = False
for path in allowed_dirs:
    if os.path.isdir(path):
        os.chdir(path)
        print(f"Working directory beállítva: {path}")
        found = True
        break
    
if not found:
    print("❌ Nem található egyik elvárt working directory sem.")
    sys.exit(1)  # Kilép a szkripttel, ha nincs meg a megfelelő könyvtár
 
output_dir = "TSDP/datafiles"
os.makedirs(output_dir, exist_ok=True)

#%% fetch data
# Alap beállítások
base_url = "https://www.football-data.co.uk/mmz4281/"
league_code = "E0"  # Premier League
seasons_to_get = 10  # Hány szezonra visszamenőleg

# Jelenlegi év (pl. 2025 augusztusában futtatva: 2025)
current_year = datetime.now().year

# Letöltés
for i in range(seasons_to_get):
    end_year = current_year - i
    start_year = end_year - 1
    # Pl. 2024-25 -> '2425'
    season_code = f"{str(start_year)[-2:]}{str(end_year)[-2:]}"
    url = f"{base_url}{season_code}/{league_code}.csv"
    output_path = os.path.join(output_dir, f"{season_code}_{league_code}.csv")

    try:
        print(f"Letöltés: {url}")
        response = requests.get(url)
        if response.status_code == 200 and "Date" in response.text:
            with open(output_path, "wb") as f:
                f.write(response.content)
            print(f"Sikeresen letöltve: {output_path}")
        else:
            print(f"⚠️ Nem sikerült letölteni vagy üres: {url}")
    except Exception as e:
        print(f"❌ Hiba történt a letöltésnél: {e}")

#%% work with data
import pandas as pd
import glob

# CSV fájlok betöltése
all_files = glob.glob("TSDP/datafiles/*_E0.csv")
all_dfs = []

for file in all_files:
    df = pd.read_csv(file)
    
    # Hozzáadjuk a szezonkódot
    season = file.split("\\")[-1].split("_")[0]  # vagy "/" ha Linux/Mac
    df["SeasonCode"] = season
    all_dfs.append(df)

# Egyesítjük őket
df_all = pd.concat(all_dfs, ignore_index=True)

# Dátum konvertálása
df_all["Date"] = pd.to_datetime(df_all["Date"], dayfirst=True, errors='coerce')

# Tisztítjuk azokat, ahol nincs eredmény
df_all = df_all.dropna(subset=["FTR", "HomeTeam", "AwayTeam"])

# Long formátum: külön sor hazai és vendégcsapatnak
home = df_all[["SeasonCode", "Date", "HomeTeam", "FTR"]].copy()
home["Team"] = home["HomeTeam"]
home["IsHome"] = True
home["Points"] = home["FTR"].map({"H": 3, "D": 1, "A": 0})

away = df_all[["SeasonCode", "Date", "AwayTeam", "FTR"]].copy()
away["Team"] = away["AwayTeam"]
away["IsHome"] = False
away["Points"] = away["FTR"].map({"A": 3, "D": 1, "H": 0})

# Egyesítjük
matches = pd.concat([home[["SeasonCode", "Date", "Team", "Points"]], 
                     away[["SeasonCode", "Date", "Team", "Points"]]],
                    ignore_index=True)

# Rendezzük időrendbe
matches = matches.sort_values(["SeasonCode", "Team", "Date"])

# Sorozatszámozás csapatonként (hanyadik meccse a szezonban)
matches["MatchNumber"] = matches.groupby(["SeasonCode", "Team"]).cumcount() + 1

# Első 10 és utolsó meccsek külön választása
first10 = matches[matches["MatchNumber"] <= 5].copy()
rest = matches[matches["MatchNumber"] > 5].copy()

# Átlagpontszám számítása
first10_avg = first10.groupby(["SeasonCode", "Team"])["Points"].mean().reset_index()
first10_avg.rename(columns={"Points": "First10_AvgPoints"}, inplace=True)

rest_avg = rest.groupby(["SeasonCode", "Team"])["Points"].mean().reset_index()
rest_avg.rename(columns={"Points": "Rest_AvgPoints"}, inplace=True)

# Összefűzés egy táblába
results = pd.merge(first10_avg, rest_avg, on=["SeasonCode", "Team"], how="outer")

# Szezonvégi összpontszám csapatonként
season_totals = matches.groupby(["SeasonCode", "Team"])["Points"].sum().reset_index()
season_totals.rename(columns={"Points": "TotalPoints"}, inplace=True)

# Rangsort generálunk minden szezonban
season_totals["FinalRank"] = season_totals.groupby("SeasonCode")["TotalPoints"]\
                                .rank(ascending=False, method="min")
# Már kiszámoltad a TotalPoints és FinalRank értékeket
season_totals["Top4"] = season_totals["FinalRank"] <= 4
season_totals["Relegated"] = season_totals["FinalRank"] >= 18
season_totals["AvgPoints"] = season_totals["TotalPoints"] / 38  # PL: 38 meccs

# Csatoljuk hozzá az eredményekhez
results = pd.merge(results, season_totals, on=["SeasonCode", "Team"], how="left")
print(results.head())

#%% ml modell
def calc_first10_stats(df_all):
    extended_cols = [
        'SeasonCode', 'Date', 'HomeTeam', 'AwayTeam',
        'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST',
        'HF', 'AF', 'HC', 'AC'
    ]
    df = df_all[extended_cols].copy()
    df["GD_Home"] = df["FTHG"] - df["FTAG"]
    df["GD_Away"] = -df["GD_Home"]

    records = []

    for season in df["SeasonCode"].unique():
        df_season = df[df["SeasonCode"] == season]
        teams = pd.unique(df_season[["HomeTeam", "AwayTeam"]].values.ravel())

        for team in teams:
            home_games = df_season[df_season["HomeTeam"] == team].copy()
            away_games = df_season[df_season["AwayTeam"] == team].copy()

            all_games = pd.concat([
                home_games.assign(IsHome=True),
                away_games.assign(IsHome=False)
            ])
            all_games = all_games.sort_values("Date").head(5)

            points = all_games.apply(
                lambda row: 3 if (row["IsHome"] and row["FTHG"] > row["FTAG"]) or
                                  (not row["IsHome"] and row["FTHG"] < row["FTAG"])
                            else (1 if row["FTHG"] == row["FTAG"] else 0),
                axis=1
            )

            stats = {
                "SeasonCode": season,
                "Team": team,
                "First10_AvgPoints": points.mean(),
                "First10_GD": (
                    all_games.apply(
                        lambda r: r["GD_Home"] if r["IsHome"] else r["GD_Away"], axis=1
                    ).sum()
                ),
                "HomeRatio_First10": all_games["IsHome"].mean(),
                "First10_ShotDiff": (
                    all_games.apply(
                        lambda r: r["HS"] - r["AS"] if r["IsHome"] else r["AS"] - r["HS"], axis=1
                    ).mean()
                ),
                "First10_ShotsOnTgt": (
                    all_games.apply(
                        lambda r: r["HST"] - r["AST"] if r["IsHome"] else r["AST"] - r["HST"], axis=1
                    ).mean()
                ),
                "First10_FoulDiff": (
                    all_games.apply(
                        lambda r: r["HF"] - r["AF"] if r["IsHome"] else r["AF"] - r["HF"], axis=1
                    ).mean()
                ),
                "First10_CornerDiff": (
                    all_games.apply(
                        lambda r: r["HC"] - r["AC"] if r["IsHome"] else r["AC"] - r["HC"], axis=1
                    ).mean()
                )
            }
            records.append(stats)
    
    return pd.DataFrame(records)

first10_features = calc_first10_stats(df_all)

# modell építése
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
import numpy as np

# Változók összekapcsolása
model_data = pd.merge(first10_features, season_totals, on=["SeasonCode", "Team"])

# Kiválasztott feature-ök
features = [
    "First10_AvgPoints", "First10_GD", "HomeRatio_First10",
    "First10_ShotDiff", "First10_ShotsOnTgt", 
    "First10_FoulDiff", "First10_CornerDiff"
]

# ---------- 1. Top4 modell ----------
X = model_data[features]
y = model_data["Top4"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf_top4 = RandomForestClassifier(n_estimators=100, random_state=42)
clf_top4.fit(X_train, y_train)

y_pred = clf_top4.predict(X_test)
print("=== Top4 Predikció ===")
print(classification_report(y_test, y_pred))

# ---------- 2. Átlagpontszám (regresszió) ----------
y_reg = model_data["AvgPoints"]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)

reg_avgpts = RandomForestRegressor(n_estimators=100, random_state=42)
reg_avgpts.fit(X_train_r, y_train_r)

y_pred_r = reg_avgpts.predict(X_test_r)
rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
print(f"=== Átlagpontszám RMSE: {rmse:.3f}")

# ---------- 3. Kiesés modell ----------
y_releg = model_data["Relegated"]

X_train_k, X_test_k, y_train_k, y_test_k = train_test_split(X, y_releg, test_size=0.2, random_state=42)

clf_kieses = RandomForestClassifier(n_estimators=100, random_state=42)
clf_kieses.fit(X_train_k, y_train_k)

y_pred_k = clf_kieses.predict(X_test_k)
print("=== Kiesés Predikció ===")
print(classification_report(y_test_k, y_pred_k))

import matplotlib.pyplot as plt

importances = clf_top4.feature_importances_
feat_importance = pd.Series(importances, index=features)
feat_importance.sort_values().plot(kind="barh")
plt.title("Feature Importance – Top4 model")
plt.tight_layout()
plt.show()

#%% analysis
# Korrelációs mátrix
print("Correlation", results[["First10_AvgPoints", "Rest_AvgPoints"]].corr().iloc[0].iloc[1])

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
sns.regplot(
    data=results,
    x="First10_AvgPoints",
    y="Rest_AvgPoints",
    line_kws={"color": "red"},
    scatter_kws={"alpha": 0.6}
)
plt.title("Első 10 meccs pontátlaga vs. hátralévő szezon pontátlaga")
plt.xlabel("Első 10 meccs – Átlagpont")
plt.ylabel("Hátralévő meccsek – Átlagpont")
plt.grid(True)
plt.tight_layout()
plt.xlim(0, 3)
plt.ylim(0, 3)
plt.show()

# Kategorizáljuk a jó rajtot
results["StrongStart"] = results["First10_AvgPoints"] >= 1.8
results["Top4Finish"] = results["FinalRank"] <= 3

# Csoportosítva: milyen arányban kerültek top 4-be?
grouped = results.groupby("StrongStart")["Top4Finish"].mean().reset_index()
print(grouped)

# Chi-négyzet teszt kontingencia táblán
from scipy.stats import chi2_contingency

contingency = pd.crosstab(results["StrongStart"], results["Top4Finish"])
chi2, p, dof, expected = chi2_contingency(contingency)

print("\nKontingenciatábla:")
print(contingency)
print(f"\nChi2 érték: {chi2:.3f}, p-érték: {p:.4f}")

if p < 0.05:
    print("✅ Szignifikáns kapcsolat: erős kezdés esetén szignifikánsan nagyobb eséllyel kerülnek top 4-be.")
else:
    print("❌ Nem szignifikáns: nincs bizonyított kapcsolat az erős kezdés és a top 4-be jutás között.")

#%% save model
import joblib

joblib.dump(clf_top4, "TSDP/strongstart/clf_top4.pkl")
joblib.dump(reg_avgpts, "TSDP/strongstart/reg_avgpts.pkl")
joblib.dump(clf_kieses, "TSDP/strongstart/clf_kieses.pkl")
joblib.dump(features, "TSDP/strongstart/features.pkl")
