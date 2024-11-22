import json
import pandas as pd
import xgboost as xgb
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the pre-trained XGBoost model
def load_model():
    try:
        model_path = "C:/new mini/xgb_model.json"  # Ensure the path is correct
        model = xgb.Booster()  # Initialize the XGBoost Booster
        model.load_model(model_path)  # Load model from file
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Load the model on server start
model = load_model()
if not model:
    raise Exception("Model could not be loaded. Please check the model path.")

# Preprocessing function to map form input to model-compatible format
def feature_engineer(data):
    # Convert data fields to the expected numerical format
    data['gender'] = 0 if data['gender'] == 'M' else 1
    data['nationality'] = int(data['nationality'])  # Map nationality accordingly
    data['stage_id'] = {'lowerlevel': 0, 'MiddleSchool': 1, 'HighSchool': 2}[data['stage_id']]
    data['grade_id'] = {'G-02': 0, 'G-04': 1, 'G-05': 2, 'G-06': 3, 'G-07': 4,
                        'G-08': 5, 'G-09': 6, 'G-10': 7, 'G-11': 8, 'G-12': 9}[data['grade_id']]
    data['section_id'] = {'A': 0, 'B': 1, 'C': 2}[data['section_id']]
    data['parent_answering_survey'] = 1 if data['parent_answering_survey'] == 'Yes' else 0
    data['parent_school_satisfaction'] = 1 if data['parent_school_satisfaction'] == 'Good' else 0
    data['student_absence_days'] = 0 if data['student_absence_days'] == 'Under-7' else 1
    return pd.DataFrame([data])

# Home route with input form
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        # Get data from form fields
        data = {
            "gender": request.form['gender'],
            "nationality": request.form['nationality'],
            "stage_id": request.form['stage_id'],
            "grade_id": request.form['grade_id'],
            "section_id": request.form['section_id'],
            "parent_answering_survey": request.form['parent_answering_survey'],
            "parent_school_satisfaction": request.form['parent_school_satisfaction'],
            "student_absence_days": request.form['student_absence_days'],
        }

        # Process the input data
        features = feature_engineer(data)

        # Make prediction
        dmatrix = xgb.DMatrix(features)
        prediction = model.predict(dmatrix)

        # Render the result page with the prediction
        return render_template('result.html', prediction=int(prediction[0]))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


