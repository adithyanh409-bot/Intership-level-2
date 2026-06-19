import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("Obesity.csv")

print("Dataset Shape:", df.shape)
print(df.head())


# ==========================================
# Create Encoders
# ==========================================

gender_encoder = LabelEncoder()
family_encoder = LabelEncoder()
favc_encoder = LabelEncoder()
caec_encoder = LabelEncoder()
smoke_encoder = LabelEncoder()
scc_encoder = LabelEncoder()
calc_encoder = LabelEncoder()
mtrans_encoder = LabelEncoder()
obesity_encoder = LabelEncoder()


# ==========================================
# Encode Categorical Columns
# ==========================================

df["Gender"] = gender_encoder.fit_transform(
    df["Gender"]
)

df["family_history_with_overweight"] = (
    family_encoder.fit_transform(
        df["family_history_with_overweight"]
    )
)

df["FAVC"] = favc_encoder.fit_transform(
    df["FAVC"]
)

df["CAEC"] = caec_encoder.fit_transform(
    df["CAEC"]
)

df["SMOKE"] = smoke_encoder.fit_transform(
    df["SMOKE"]
)

df["SCC"] = scc_encoder.fit_transform(
    df["SCC"]
)

df["CALC"] = calc_encoder.fit_transform(
    df["CALC"]
)

df["MTRANS"] = mtrans_encoder.fit_transform(
    df["MTRANS"]
)

df["NObeyesdad"] = obesity_encoder.fit_transform(
    df["NObeyesdad"]
)


# ==========================================
# Save Encoders
# ==========================================

encoders = {
    "Gender": gender_encoder,
    "family_history_with_overweight": family_encoder,
    "FAVC": favc_encoder,
    "CAEC": caec_encoder,
    "SMOKE": smoke_encoder,
    "SCC": scc_encoder,
    "CALC": calc_encoder,
    "MTRANS": mtrans_encoder,
    "NObeyesdad": obesity_encoder
}

joblib.dump(encoders, "encoders.pkl")


# ==========================================
# Features & Target
# ==========================================

X = df.drop("NObeyesdad", axis=1)

y = df["NObeyesdad"]


# ==========================================
# Feature Scaling
# ==========================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, "scaler.pkl")


# ==========================================
# Train/Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# Build ANN Model
# ==========================================

model = Sequential()

model.add(
    Dense(
        32,
        activation="relu",
        input_shape=(X_train.shape[1],)
    )
)

model.add(
    Dense(
        16,
        activation="relu"
    )
)

model.add(
    Dense(
        8,
        activation="relu"
    )
)

model.add(
    Dense(
        7,
        activation="softmax"
    )
)


# ==========================================
# Compile Model
# ==========================================

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# Train Model
# ==========================================

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.10,
    verbose=1
)


# ==========================================
# Evaluate Model
# ==========================================

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(f"\nAccuracy: {accuracy * 100:.2f}%")
print(f"Loss: {loss:.4f}")


# ==========================================
# Save Model
# ==========================================

model.save("obesity_ann_model.h5")

print("\nFiles Saved Successfully")
print("✔ obesity_ann_model.h5")
print("✔ scaler.pkl")
print("✔ encoders.pkl")


# ==========================================
# Display Encoder Classes
# ==========================================

print("\nObesity Classes:")

for index, label in enumerate(
    obesity_encoder.classes_
):
    print(f"{index} -> {label}")