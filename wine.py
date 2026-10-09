
from flask import Flask, request, render_template_string, send_from_directory
import numpy as np
import joblib

app = Flask(__name__)

ans = joblib.load("wine.joblib")
model = ans["model"]
cols = ans["columns"]


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    values = {}

    if request.method == "POST":
        try:
            res = []

            for col in cols:
                value = float(request.form[col])
                res.append(value)
                values[col] = value

            prediction = model.predict(np.array([res]))[0]

            if isinstance(prediction, np.generic):
                prediction = prediction.item()

        except (ValueError, TypeError):
            error = "Please enter valid numeric values."
        except Exception:
            error = "Prediction failed. Please check your inputs."

    with open("wine.html", "r", encoding="utf-8") as file:
        html = file.read()

    return render_template_string(
        html,
        cols=cols,
        prediction=prediction,
        error=error,
        values=values
    )


@app.route("/wines.css")
def css():
    return send_from_directory(".", "wines.css")


if __name__ == "__main__":
    app.run(debug=True)
