import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Générer des données factices pour illustrer l'exemple
import numpy as np
data = np.random.random((1000, 10))  # 1000 séquences de 10 valeurs

# Préparer les données en séquences
seq_length = 5
X, y = [], []
for i in range(len(data) - seq_length):
    seq = data[i:i + seq_length]
    label = data[i + seq_length]
    X.append(seq)
    y.append(label)

X = np.array(X)
y = np.array(y)

# Définir le modèle LSTM
model = Sequential()
model.add(LSTM(50, input_shape=(seq_length, 10)))
model.add(Dense(10))  # La couche de sortie a 10 neurones car nos séquences ont 10 valeurs

# Compiler le modèle
model.compile(optimizer='adam', loss='mse')  # mse pour la régression

# Entraîner le modèle
model.fit(X, y, epochs=10, batch_size=32)
