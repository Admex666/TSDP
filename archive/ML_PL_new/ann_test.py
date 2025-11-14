import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.utils import compute_class_weight

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
y = model_input.loc[:, 'O/U2.5']
y_map = y.map({'Over': 1, 'Under':0})
#y_map = y.map({'H': 0, 'A': 2, 'D': 1})

# Adatok normalizálása és train-test split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Feature scaling

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_map, test_size=0.2, random_state=600)

# Neurális háló építése
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(2, activation='softmax')
])

# Modell fordítása
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#%% 4. HIPERPARAMÉTEREK BEÁLLÍTÁSA
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Class weights számítása (kiegyensúlyozatlan adatokhoz)
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights = dict(enumerate(class_weights))

#%% 5. MODELLTANÍTÁS FEJLESZTÉSE
history = model.fit(
    X_train, 
    y_train,
    epochs=100,  # Több epoch, de early stopping
    batch_size=64,
    validation_split=0.25,
    class_weight=class_weights,
    callbacks=[early_stop]
)

#%% 6. RÉSZLETES KIÉRTÉKELÉS
from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test).argmax(axis=1)
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))