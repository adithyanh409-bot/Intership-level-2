import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import joblib

# Load Dataset
df = pd.read_csv("heart.csv")

print("First 5 Records")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

# Label Encoding
le = LabelEncoder()
df["thal"] = le.fit_transform(df["thal"])

# Input and Output
X = df.drop("target", axis=1)
Y = df["target"]

# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build ANN Model
model = Sequential()

# Input Layer + Hidden Layer 1
model.add(Dense(
    16,
    activation='relu',
    input_shape=(X_train.shape[1],)
))

# Hidden Layer 2
model.add(Dense(
    8,
    activation='relu'
))

# Hidden Layer 3
model.add(Dense(
    4,
    activation='relu'
))

# Output Layer
model.add(Dense(
    1,
    activation='sigmoid'
))

# Compile Model
model.compile(
    optimizer=Adam(),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train Model
model.fit(
    X_train,
    Y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.1
)

# Prediction
y_prob = model.predict(X_test)

y_pred = (y_prob > 0.5).astype(int)

# Accuracy
print("\nAccuracy Score:")
print(accuracy_score(Y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(Y_test, y_pred))

# Save Model
model.save("heart_ann_model.h5")

# Save Scaler
joblib.dump(scaler, "heart_scaler.pkl")

print("\nModel and Scaler Saved Successfully")