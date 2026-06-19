from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load trained assets
model = load_model("models/obesity_ann_model.h5")
scaler = joblib.load("models/scaler.pkl")
encoders = joblib.load("models/encoders.pkl")

obesity_encoder = encoders["NObeyesdad"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        gender = encoders["Gender"].transform(
            [data["Gender"]]
        )[0]

        family = encoders[
            "family_history_with_overweight"
        ].transform(
            [data["family_history_with_overweight"]]
        )[0]

        favc = encoders["FAVC"].transform(
            [data["FAVC"]]
        )[0]

        caec = encoders["CAEC"].transform(
            [data["CAEC"]]
        )[0]

        smoke = encoders["SMOKE"].transform(
            [data["SMOKE"]]
        )[0]

        scc = encoders["SCC"].transform(
            [data["SCC"]]
        )[0]

        calc = encoders["CALC"].transform(
            [data["CALC"]]
        )[0]

        mtrans = encoders["MTRANS"].transform(
            [data["MTRANS"]]
        )[0]

        features = np.array([[
            gender,
            data["Age"],
            data["Height"],
            data["Weight"],
            family,
            favc,
            data["FCVC"],
            data["NCP"],
            caec,
            smoke,
            data["CH2O"],
            scc,
            data["FAF"],
            data["TUE"],
            calc,
            mtrans
        ]])

        features = scaler.transform(features)

        prediction = model.predict(features)

        predicted_class = np.argmax(
            prediction,
            axis=1
        )[0]

        obesity_level = obesity_encoder.inverse_transform(
            [predicted_class]
        )[0]

        confidence = float(
            np.max(prediction) * 100
        )

        return jsonify({
            "success": True,
            "prediction": obesity_level,
            "confidence": round(confidence, 2)
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        })


if __name__ == "__main__":
    app.run(debug=True)