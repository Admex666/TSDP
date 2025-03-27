import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

import os
wd_old = os.getcwd()
if wd_old != 'C:\\Users\\Adam\\..Data\\TSDP':
    wd_base = wd_old.split('\\')[:4]
    wd_new = '\\'.join(wd_base)+'\\TSDP'
    os.chdir(wd_new)
from ML_PL_new.ML_PL_transform_data import df_to_model_input 

#%% Loading data from website
url22 = "https://www.football-data.co.uk/mmz4281/2223/E0.csv"
url23 = "https://www.football-data.co.uk/mmz4281/2324/E0.csv"
df22 = pd.read_csv(url22)
df23 = pd.read_csv(url23)
df = pd.concat([df22, df23]).reset_index()
# Only needed columns
needed_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HY', 'AY', 'FTR']
betting_cols = ['B365H', 'B365D', 'B365A', 'B365>2.5', 'B365<2.5']
df = df[needed_cols+betting_cols]
# create BTTS and O2,5 labels
df['BTTS'] = np.where((df.FTHG!=0)&(df.FTAG!=0),'Yes','No')
df['O/U2.5'] = np.where(df.FTHG+df.FTAG>2.5,'Over','Under')

model_input = df_to_model_input(df, weather=False)

#%%
X = model_input.iloc[:,6:]
y = model_input.loc[:, 'FTR']


# Adatok normalizálása és train-test split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Feature scaling

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=600)

# Neurális háló építése
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),  # Rejtett réteg 1
    Dropout(0.3),  # Overfitting elkerülése
    Dense(16, activation='relu'),  # Rejtett réteg 2
    Dense(3, activation='softmax')
])

# Modell fordítása
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Modell tanítása
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)

# Modell kiértékelése
loss, acc = model.evaluate(X_test, y_test)
gnb.evaluate(X_test, y_test)
print(f"Accuracy: {acc:.4f}")
