import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("lab_11_bridge_data.csv")

# Encode categorical feature 'Material'
label_encoder = LabelEncoder()
data['Material'] = label_encoder.fit_transform(data['Material'])

# Define features and target
X = data.drop(columns=["Bridge_ID", "Max_Load_Tons"])
y = data["Max_Load_Tons"]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save preprocessing pipeline
with open("preprocessing.pkl", "wb") as f:
    pickle.dump((scaler, label_encoder), f)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build ANN model
model = Sequential([
    Dense(64, activation='relu', kernel_regularizer=l2(0.01), input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    Dense(1)
])

# Compile model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=16, callbacks=[early_stopping])

# Save model
model.save("tf_bridge_model.h5")

# Plot training history
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Validation Loss")
plt.show()
