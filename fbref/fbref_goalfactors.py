import pandas as pd
from TSDP.fbref import fbref_module as fbref

if 'df_raw' not in locals():
    #df_raw = fbref.get_all_team_data('Big5', year=False)
    path = 'TSDP/fbref/Big5_teamdata.csv'
    #df_raw.to_csv(path, index=False)
    df_raw = pd.read_csv(path)
stats_dict = fbref.stats_dict()

#%% formatting, dropping cols which are tied to goals
df = df_raw.copy()

# new variables
df['xG per Shot on Target'] = df['Expected_xG'] / df['Standard_SoT']
df['Deep pass ratio %'] = (df['PPA'] + df['CrsPA']) / df['Att']
df['Touches Box share %'] = df['Touches_Att Pen'] / df['Touches_Touches']
df['Progressive Pass ratio %'] = df['PrgP'] / df['Att']

cols_drop = ['Performance_W', 'Performance_D', 'Performance_L', 'Performance_CS', 'Performance_CS%'] +\
            [col for col in df.columns if 'Goals_' in col] +\
            [col for col in df.columns if 'Playing Time_' in col] +\
            [col for col in df.columns if 'Expected_' in col] +\
            [col for col in df.columns if ('xG' in col) and (col != 'xG per Shot on Target')] +\
            [col for col in df.columns if 'xAG' in col] +\
            [col for col in df.columns if 'G+A' in col] +\
            [col for col in df.columns if '_G/S' in col] +\
            [col for col in df.columns if ('Gls' in col) and (col != 'Performance_Gls')] +\
            [col for col in df.columns if '_GA' in col] +\
            [col for col in df.columns if 'Ast' in col] +\
            [col for col in df.columns if '-PK' in col] +\
            [col for col in df.columns if 'GCA' in col] + ['Standard_SoT/90']
df.drop(columns=cols_drop, inplace=True)

cols_basic = ['Rk', 'Squad', 'Comp', '# Pl', 'Age', '90s']
cols_num = [col for col in df.columns if col not in cols_basic]

df90 = df.copy()
for col in cols_num:
    if (col != 'Poss') and ('%' not in col) and ('90' not in col) \
        and ('_G/S' not in col) and (col != 'xG per Shot on Target'):
        df90[col] = df90[col] / df90['90s']

state = 111
#%% What influences the number of goals for and goals against: Linear regr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

target = 'Performance_Gls'
X = df90[[col for col in cols_num if col != target]]
y = df90[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_scaled, y, test_size=0.25, random_state=state)

regr_s = LinearRegression()
regr_s.fit(X_train_s, y_train_s)
print(f'Score: {regr_s.score(X_test_s, y_test_s):.3f}\n')

# Feature importance újra, skálázott adatokkal
coef_df_scaled = pd.DataFrame({'Feature': X.columns, 'Coefficient': regr_s.coef_})
coef_df_scaled = coef_df_scaled.sort_values(by='Coefficient', ascending=False)
print(coef_df_scaled)
print('')

from sklearn.model_selection import cross_val_score
import numpy as np

scores = cross_val_score(regr_s, X_scaled, y, cv=5, scoring='r2')

print(f'Cross-validated R^2 scores: {scores}')
print(f'Mean R^2: {np.mean(scores):.3f} | Std: {np.std(scores):.3f}')

#%% Random forest regr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Célváltozó és prediktorok
pred_col = 'Performance_Gls'
X = df90[[col for col in cols_num if col != pred_col ]]
y = df90[pred_col]

# Nem kötelező, de ha más modellekhez is használsz skálázást:
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Random Forest modell
rf = RandomForestRegressor(n_estimators=100, random_state=state)

# Cross-validation
cv_scores = cross_val_score(rf, X_scaled, y, cv=5, scoring='r2')

print(f'Cross-validated R² scores: {cv_scores}')
print(f'Mean R²: {np.mean(cv_scores):.3f} | Std: {np.std(cv_scores):.3f}')

# Modell betanítása az egész adaton a feature importance-hoz
rf.fit(X_scaled, y)

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)
for i in feature_importance.index:
    stat = feature_importance.loc[i, 'Feature']
    feature_importance.loc[i, 'Feature_pretty'] = stats_dict[stat]['name'] if stat in stats_dict.keys() else None

#for i in feature_importance[pd.isna(feature_importance.Feature_pretty)].index:
#    print(feature_importance.loc[i, 'Feature'])

print('\nTop 15 Important Features:')
print(feature_importance[['Feature_pretty', 'Importance']].head(15))

#%% Partial Dependence plot
rf_raw = RandomForestRegressor(random_state=state)
rf_raw.fit(X, y)

from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

features = ['Standard_SoT', 'PPA', 'Touches_Att Pen']
PartialDependenceDisplay.from_estimator(rf_raw, X, features, feature_names=X.columns, random_state=state)
plt.tight_layout()
plt.show()

#%% ICE plot
features = ['Standard_SoT', 'PPA', 'Touches_Att Pen']

# ICE plot
PartialDependenceDisplay.from_estimator(
    rf_raw,          # modell (nem skálázott, eredeti tanított RF)
    X,               # bemenet (eredeti, nem skálázott!)
    features,        # vizsgált jellemzők
    feature_names=X.columns,
    kind="individual",   # <- ez teszi ICE plot-tá
    subsample=20,        # ne az összes vonalat rajzolja (20 véletlenszerű sor)
    grid_resolution=100, # mennyi pontból álljon a vonal
    random_state=state
)
plt.tight_layout()
plt.show()

#%% Get extarnal data
df_hun = fbref.get_all_team_data_huv('HUN')
df_hun = df_hun.dropna(axis=1)

#%% Test the ML model on external data
df_hun90 = df_hun.copy()
for col in cols_num:
    if col in df_hun90.columns:
        if (col != 'Poss') and ('%' not in col) and ('90' not in col) \
            and ('_G/S' not in col) and (col != 'xG per Shot on Target'):
            df_hun90[col] = df_hun90[col] / df_hun90['90s']

# Build Random Forest
features = [col for col in cols_num if (col in df_hun90.columns) and (col != 'Performance_Gls')]

X_train = df90[features]
y_train = df90[target]

X_hun = df_hun90[features]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_hun_scaled = scaler.transform(X_hun)

rf = RandomForestRegressor(n_estimators=100, random_state=state)
rf.fit(X_train_scaled, y_train)

from scipy.stats import norm
# 90%-os confidence interval
z = norm.ppf(0.95)  # 1.64
sigma = 0.05 

preds = rf.predict(X_hun_scaled)

# Konfidencia intervallum
lower_bound = preds - z * sigma
upper_bound = preds + z * sigma

# Hozzáadás a DataFrame-hez
df_hun90['Predicted_Gls'] = preds
df_hun90['CI90_Lower'] = lower_bound
df_hun90['CI90_Upper'] = upper_bound

team_preds = df_hun90.groupby('Squad')['Predicted_Gls'].mean().sort_values(ascending=False)
print(team_preds)

MP = 33
for i, row in df_hun90.iterrows():
    (team, goals90_fact, goals90_pred, goals90_pred_lower, 
     goals90_pred_upper) = row[['Squad', target, 'Predicted_Gls', 'CI90_Lower', 'CI90_Upper']]
    
    is_in_interval = (goals90_pred_lower < goals90_fact) and (goals90_pred_upper > goals90_fact)
    
    print(f'{team} scored {goals90_fact*MP:.0f} (pred: {goals90_pred*MP:.0f}).\nPrediction was over {goals90_pred_lower*MP:.0f}, and under {goals90_pred_upper*MP:.0f}.')
    print('Prediction was in the interval.') if is_in_interval else print('Prediction was NOT in the interval.')
    print('')

abs_error = sum(abs(df_hun90['Predicted_Gls'] - df_hun90[target]))*MP
print(f'Absoloute error of prediction: {abs_error:.1f} goals\n{abs_error/len(df_hun90):.1f} goals on average.')
