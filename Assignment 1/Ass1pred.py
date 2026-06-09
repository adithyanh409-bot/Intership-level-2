import joblib
import pandas as pd

model = joblib.load("model.pkl")
x_scaler = joblib.load("x_scaler.pkl")
y_scaler = joblib.load("y_scaler.pkl")

sex_encoder = joblib.load("sex_encoder.pkl")
smoker_encoder = joblib.load("smoker_encoder.pkl")
region_encoder = joblib.load("region_encoder.pkl")

sex = sex_encoder.transform(["male"])[0]
smoker = smoker_encoder.transform(["no"])[0]
region = region_encoder.transform(["southeast"])[0]

new_data = pd.DataFrame([[18, sex, 33.77, 1, smoker, region]],columns=["age","sex","bmi","children","smoker","region"])

new_data_scaled = x_scaler.transform(new_data)

predicted_scaled = model.predict(new_data_scaled)

predicted_charge = y_scaler.inverse_transform(predicted_scaled)

print(predicted_charge[0][0])