from flask import Flask, render_template, request
import pickle
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print('yup!')
    if request.method == 'POST' :
        gender = request.form.get('gender')
        age = int(request.form.get('age'))
        driving_license = int(request.form.get('driving_license'))
        region_code = int(request.form.get('region_code'))
        previously_insured = int(request.form.get('previously_insured'))
        vehicle_age = request.form.get('vehicle_age')
        vehicle_damage = request.form.get('vehicle_damage')
        annual_premium = float(request.form.get('annual_premium'))
        policy_sales_channel = int(request.form.get('policy_sales_channel'))
        vintage = int(request.form.get('vintage'))

        arr = [gender, age, driving_license, region_code, previously_insured,
                vehicle_age, vehicle_damage, annual_premium, policy_sales_channel, vintage]

        scaled_arr = scaler.transform([arr])

        # Make prediction
        prediction = model.predict_proba(scaled_arr)
        res = ''
        if prediction[0][0] > 0.5 :
            res = 'The response is probably a no with the probability of ' + str(round(prediction[0][0], 2))
        else :
            res = 'The response is probably a yes with the probability of  ' + str(round(prediction[0][1], 2))

        # Render results.html with the prediction
        return render_template('results.html', prediction=res)
    
    return render_template('prediction-page.html')

if __name__ == "__main__":
    app.run(debug=True)
