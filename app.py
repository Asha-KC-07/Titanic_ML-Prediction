import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('titanic_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    try:
        # Get form data
        pclass = int(request.form['Pclass'])
        age = float(request.form['Age'])
        sibsp = int(request.form['SibSp'])
        parch = int(request.form['Parch'])
        fare = float(request.form['Fare'])
        sex = request.form['Sex']  # Expected to be 'male' or 'female'
        embarked = request.form['Embarked']  # Expected values: 'C', 'Q', or 'S'

        # Create a DataFrame for the input features
        input_df = pd.DataFrame({
            'Pclass': [pclass],
            'Age': [age],
            'SibSp': [sibsp],
            'Parch': [parch],
            'Fare': [fare],
            'Sex': [sex],
            'Embarked': [embarked]
        })
        
        # Preprocess input similar to training
        # (Fill missing values if necessary; here we assume values are provided)
        input_df = pd.get_dummies(input_df, columns=['Sex', 'Embarked'], drop_first=True)
        
        # We expect the same features as used during training:
        # ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
        expected_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
        for col in expected_cols:
            if col not in input_df.columns:
                input_df[col] = 0  # Add missing binary columns with default value zero
        # Ensure columns are in the same order as expected
        input_df = input_df[expected_cols]
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_text = "Survived" if prediction == 1 else "Did not survive"
        
        # Render result on the same page (you can customize this as needed)
        return render_template('index.html', prediction_text=f"Prediction: {prediction_text}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

#if __name__ == "__main__":
#    app.run(debug=True)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if $PORT is not set
    app.run(debug=True, host='0.0.0.0', port=port)

