import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# Load the dataset
data = pd.read_csv('lab_11_bridge_data.csv')

# Handle missing values for numerical features only
numerical_features = ['Span_ft', 'Deck_Width_ft', 'Age_Years', 'Num_Lanes', 'Condition_Rating']
data[numerical_features] = data[numerical_features].fillna(data[numerical_features].mean())

# One-hot encode the 'Material' categorical variable
data = pd.get_dummies(data, columns=['Material'])

# Normalize/standardize numerical features
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Ensure all features are numeric
X = data.drop(columns=['Max_Load_Tons'])
y = data['Max_Load_Tons']

# Convert all columns to numeric, forcing errors to NaN
X = X.apply(pd.to_numeric, errors='coerce')

# Handle any remaining missing values
X.fillna(0, inplace=True)

# Convert DataFrame to NumPy array
X = X.to_numpy()
y = y.to_numpy()

# Ensure all elements in the arrays are of type float
X = X.astype(np.float32)
y = y.astype(np.float32)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the ANN model
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    Dense(1, activation='linear')
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Plot training/validation loss vs. epochs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save the model
model.save('tf_bridge_model.h5')
