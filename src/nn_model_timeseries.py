# src/nn_model_timeseries.py
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from features import get_data
import matplotlib.pyplot as plt

# --- ÚJ IMPORTOK A RÉSZLETES KIÉRTÉKELÉSHEZ ---
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# --- 1. Adatbetöltés IDŐSOROS módban ---
WINDOW_SIZE = 50
print(f"🚀 Adatok betöltése 'timeseries' módban (Ablak: {WINDOW_SIZE})...")
X_train, X_test, y_train, y_test, le = get_data(mode='timeseries', window_size=WINDOW_SIZE)

# --- 2. Skálázás (3D adatokhoz) ---
print("⏳ Adatok skálázása (3D)...")
n_samples_train, n_timesteps, n_features = X_train.shape
n_samples_test = X_test.shape[0]
X_train_2d = X_train.reshape(-1, n_features)
scaler = StandardScaler()
X_train_scaled_2d = scaler.fit_transform(X_train_2d)
X_train_scaled = X_train_scaled_2d.reshape(n_samples_train, n_timesteps, n_features)
X_test_2d = X_test.reshape(-1, n_features)
X_test_scaled_2d = scaler.transform(X_test_2d)
X_test_scaled = X_test_scaled_2d.reshape(n_samples_test, n_timesteps, n_features)

# --- 3. Címkék előkészítése ---
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# --- 4. A Conv1D Modell Építése ---
print("🧠 Conv1D modell építése...")
model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)),
    MaxPooling1D(pool_size=2),
    Conv1D(64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(100, activation='relu'),
    Dropout(0.5),
    Dense(y_train_cat.shape[1], activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary() 

# --- 5. Modell Tanítása ---
print("⏳ Modell tanítása...")
history = model.fit(
    X_train_scaled, 
    y_train_cat, 
    validation_data=(X_test_scaled, y_test_cat),
    epochs=20,
    batch_size=32,
    verbose=1
)

# --- 6. KIÉRTÉKELÉS (BŐVÍTETT VERZIÓ) ---
print("\n" + "="*30)
print("📊 Modell Részletes Kiértékelése...")
print("="*30)

# Alap kiértékelés (a 'loss' és 'accuracy' értékekhez)
test_loss, test_acc = model.evaluate(X_test_scaled, y_test_cat)
print(f"Alap kiértékelés (Accuracy): {test_acc:.6f}")
print(f"Alap kiértékelés (Loss): {test_loss:.6f}")

# A modell predikcióinak kinyerése
y_pred_probs = model.predict(X_test_scaled) # Pl. [0.1, 0.9]
y_pred = np.argmax(y_pred_probs, axis=1) # Pl. 1

# A valós címkék (y_test) már a helyes formátumban vannak (pl. 0, 1, 0...)

target_names = le.classes_ # ['unworn', 'worn']

# 1. RÉSZLETES OSZTÁLYOZÁSI RIPORT
print("\n--- Osztályozási Riport ---")
print(classification_report(y_test, y_pred, target_names=target_names))

# 2. KONFÚZIÓS MÁTRIX
print("📊 Konfúziós Mátrix generálása...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names)
plt.title(f'Conv1D Konfúziós Mátrix (Ablak: {WINDOW_SIZE})', fontsize=16)
plt.ylabel('Valós Címke (True Label)', fontsize=12)
plt.xlabel('Jósolt Címke (Predicted Label)', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix_timeseries.png')
plt.close()
print(f"✅ 'confusion_matrix_timeseries.png' mentve.")

# 3. TANULÁSI GÖRBÉK (ez már megvolt)
print("📊 Tanulási görbék generálása...")
pd.DataFrame(history.history).plot(figsize=(10, 6))
plt.grid(True)
plt.gca().set_ylim(0, 1.05) # Kicsit 1 fölé engedjük a loss miatt
plt.title(f'Conv1D Tanulási Görbék (Ablak: {WINDOW_SIZE})', fontsize=16)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Érték', fontsize=12)
plt.tight_layout()
plt.savefig('learning_curves_timeseries.png')
plt.close()
print(f"✅ 'learning_curves_timeseries.png' mentve.")