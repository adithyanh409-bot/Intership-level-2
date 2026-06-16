import numpy as np
import joblib

from tensorflow.keras.models import load_model

model = load_model("heart_ann_model.h5")
scaler = joblib.load("heart_scaler.pkl")

sample = np.array([[63,1,3,145,233,1,0,150,0,2.3,0,0,1]])

sample = scaler.transform(sample)

probability = model.predict(sample)

print("Prediction Probability =", probability[0][0])

if probability[0][0] >= 0.5:
    print("Heart Disease Detected")
else:
    print("No Heart Disease")