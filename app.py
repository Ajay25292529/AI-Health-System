from flask import Flask, render_template, request
import joblib
from utils.database import init_db, insert_record, fetch_all
from utils.preprocessing import preprocess_input

app = Flask(__name__)

init_db()

# Load models
diabetes_model = joblib.load("models/diabetes_model.pkl")
diabetes_scaler = joblib.load("models/diabetes_scaler.pkl")

heart_model = joblib.load("models/heart_model.pkl")
heart_scaler = joblib.load("models/heart_scaler.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    name = request.form["name"]
    disease = request.form["disease"]

    if disease == "diabetes":
        data = [
            float(request.form["pregnancies"]),
            float(request.form["glucose"]),
            float(request.form["bp"]),
            float(request.form["skin"]),
            float(request.form["insulin"]),
            float(request.form["bmi"]),
            float(request.form["dpf"]),
            float(request.form["age"])
        ]
        scaled = diabetes_scaler.transform([data])
        prediction = diabetes_model.predict(scaled)[0]
        result = "Diabetic" if prediction == 1 else "Not Diabetic"

    else:
        data = [float(x) for x in request.form.getlist("heart_values")]
        scaled = heart_scaler.transform([data])
        prediction = heart_model.predict(scaled)[0]
        result = "Heart Disease" if prediction == 1 else "Healthy"

    insert_record(name, disease, result)

    return render_template("result.html", result=result)

@app.route("/dashboard")
def dashboard():
    records = fetch_all()
    return render_template("dashboard.html", records=records)

if __name__ == "__main__":
    app.run(debug=True)
