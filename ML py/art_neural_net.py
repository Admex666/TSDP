import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.naive_bayes import GaussianNB

# 1️⃣ Adatok betöltése
digits = load_digits()
X, y = digits.data, digits.target

# 2️⃣ Adatok normalizálása és train-test split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Feature scaling

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3️⃣ Neurális háló építése
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Rejtett réteg 1
    Dropout(0.3),  # Overfitting elkerülése
    Dense(32, activation='relu'),  # Rejtett réteg 2
    Dense(10, activation='softmax')  # Kimeneti réteg (10 számjegy: 0-9)
])
gnb = GaussianNB()

# 4️⃣ Modell fordítása
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 5️⃣ Modell tanítása
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)
cross_val_score(gnb, X_scaled, y, n_jobs=5, scoring='accuracy')

# 6️⃣ Modell kiértékelése
loss, acc = model.evaluate(X_test, y_test)
gnb.evaluate(X_test, y_test)
print(f"Teszt pontosság: {acc:.4f}")

# 7️⃣ Egy mintapélda megtekintése és predikció
sample = X_test[2].reshape(1, -1)
prediction = np.argmax(model.predict(sample))

plt.imshow(X_test[2].reshape(8, 8), cmap='gray')
plt.title(f"Predikció: {prediction}")
plt.show()
