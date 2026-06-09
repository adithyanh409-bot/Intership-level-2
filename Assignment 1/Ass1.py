import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib

data = pd.read_csv("insurance.csv")

sex_encoder = LabelEncoder()
smoker_encoder = LabelEncoder()
region_encoder = LabelEncoder()

data["sex"] = sex_encoder.fit_transform(data["sex"])
data["smoker"] = smoker_encoder.fit_transform(data["smoker"])
data["region"] = region_encoder.fit_transform(data["region"])

X = data.drop("charges", axis=1)
Y = data[["charges"]]

x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_scaled = x_scaler.fit_transform(X)
Y_scaled = y_scaler.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)

print("RMSE =", rmse)

joblib.dump(model, "model.pkl")
joblib.dump(x_scaler, "x_scaler.pkl")
joblib.dump(y_scaler, "y_scaler.pkl")

joblib.dump(sex_encoder, "sex_encoder.pkl")
joblib.dump(smoker_encoder, "smoker_encoder.pkl")
joblib.dump(region_encoder, "region_encoder.pkl")