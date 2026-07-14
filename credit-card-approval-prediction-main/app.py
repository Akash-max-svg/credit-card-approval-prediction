from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# ===========================
# Load Trained Model
# ===========================
model = joblib.load("credit_card_model.pkl")

# If you used StandardScaler during training,
# uncomment the next line and use scaler.transform()
# scaler = joblib.load("scaler.pkl")


# ===========================
# Home Page
# ===========================
@app.route("/")
def home():
    return render_template("home.html")


# ===========================
# Prediction Form
# ===========================
@app.route("/predict")
def predict():
    return render_template("index.html")


# ===========================
# Result
# ===========================
@app.route("/result", methods=["POST"])
def result():

    try:

        # -----------------------------
        # Collect Input Values
        # -----------------------------

        gender = int(request.form["gender"])
        own_car = int(request.form["own_car"])
        own_realty = int(request.form["own_realty"])

        income = float(request.form["income"])

        income_type = int(request.form["income_type"])

        education = int(request.form["education"])

        family_status = int(request.form["family_status"])

        housing_type = int(request.form["housing_type"])

        occupation = int(request.form["occupation"])

        age = float(request.form["age"])
        experience = float(request.form["experience"])

        # Convert to the format expected by the model
        days_birth = age * 365
        if income_type == 1:          # Pensioner
            days_employed = 365243
        else:
            days_employed = experience * 365

        children = int(request.form["children"])

        family_members = float(request.form["family_members"])
        credit_window = float(request.form["window"])


        # -------------------------------------
        # Create DataFrame
        # Feature order MUST match training
        # -------------------------------------


        data = pd.DataFrame([[
            gender,
            own_car,
            own_realty,
            children,
            income,
            income_type,
            education,
            family_status,
            housing_type,
            days_birth,
            days_employed,
            occupation,
            family_members,
            credit_window
        ]],columns=[
            "CODE_GENDER",
            "FLAG_OWN_CAR",
            "FLAG_OWN_REALTY",
            "CNT_CHILDREN",
            "AMT_INCOME_TOTAL",
            "NAME_INCOME_TYPE",
            "NAME_EDUCATION_TYPE",
            "NAME_FAMILY_STATUS",
            "NAME_HOUSING_TYPE",
            "DAYS_BIRTH",
            "DAYS_EMPLOYED",
            "OCCUPATION_TYPE",
            "CNT_FAM_MEMBERS",
            "window"
        ])

        # --------------------------
        # Scale Data if Required
        # --------------------------

        # data = scaler.transform(data)

        prediction = model.predict(data)[0]

        if prediction == 1:
            prediction_text = "Credit Card Approved"
            status = "approved"

        else:
            prediction_text = "Credit Card Rejected"
            status = "rejected"

        return render_template(
            "result.html",
            prediction=prediction_text,
            status=status
        )

    except Exception as e:
        return render_template(
            "result.html",
            prediction="Error : " + str(e),
            status="error"
        )


# ===========================
# Run Flask
# ===========================

if __name__ == "__main__":
    app.run(debug=True)