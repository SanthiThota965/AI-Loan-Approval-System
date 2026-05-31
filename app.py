import sqlite3
from flask import Flask, render_template, request
import numpy as np
import pickle

# Flask app
app = Flask(__name__)

# Load ML model
model = pickle.load(open('model.pkl', 'rb'))


# Home Page
@app.route('/')
def home():
    return render_template('index.html')


# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():

    # Get form values

    customer_name = request.form['Customer_Name']

    email = request.form['Email']

    gender = request.form['Gender']

    married = request.form['Married']
    
    dependents = request.form['Dependents']

    education = request.form['Education']
    
    self_employed = request.form['Self_Employed']

    applicant_income = float(
        request.form['ApplicantIncome']
    )

    coapplicant_income = float(
        request.form['CoapplicantIncome']
    )

    loan_amount = float(
        request.form['LoanAmount']
    )

    loan_term = float(
        request.form['Loan_Amount_Term']
    )

    credit_history = float(
        request.form['Credit_History']
    )

    property_area = request.form['Property_Area']

    # Convert categorical values

    gender = 1 if gender == 'Male' else 0

    married = 1 if married == 'Yes' else 0
    
    dependents = 3 if dependents == '3' else int(dependents)

    education = 1 if education == 'Graduate' else 0
    
    self_employed = 1 if self_employed == 'Yes' else 0

    property_area_map = {
        'Rural': 0,
        'Semiurban': 1,
        'Urban': 2
    }

    property_area = property_area_map[property_area]

    # Prepare model input

    data = np.array([[

    gender,
    married,
    dependents,
    education,
    self_employed,
    applicant_income,
    coapplicant_income,
    loan_amount,
    loan_term,
    credit_history,
    property_area

]])
    
    if credit_history == 0:

        result = "Rejected"
        approval_percent = 10

    elif applicant_income < 2000:

        result = "Rejected"
        approval_percent = 15

    elif applicant_income < 3000 and loan_amount > 150:

        result = "Rejected"
        approval_percent = 20

    elif applicant_income < 5000 and loan_amount > 250:

        result = "Rejected"
        approval_percent = 25

    else:

        prediction = model.predict(data)

        probability = model.predict_proba(data)

        approval_percent = round(
            probability[0][1] * 100,
            2
        )

        if prediction[0] == 1:
            result = "Approved"
        else:
            result = "Rejected"
    
    # Save into database

    conn = sqlite3.connect("loan_applications.db")

    cursor = conn.cursor()

    cursor.execute("""

    INSERT INTO applications (

        customer_name,
        email,
        gender,
        married,
        education,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_history,
        property_area,
        prediction

    )

    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

    """, (

        customer_name,
        email,
        request.form['Gender'],
        request.form['Married'],
        request.form['Education'],
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_history,
        request.form['Property_Area'],
        result

    ))

    conn.commit()
    
    print("Data inserted successfully")

    conn.close()

    # Return result page

    return render_template(
        'result.html',
        prediction=result,
        approval_percent=approval_percent
    )


# Run Flask App
if __name__ == '__main__':
    app.run(debug=True)