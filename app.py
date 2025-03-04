from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load the trained model
model_path = os.path.join('models', 'car_price_prediction_model.pkl')
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    # Create a simple fallback model for demonstration
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=10)
    # Train with dummy data so it can make predictions
    model.fit(
        [[2020, 50000, 2.5, 0, 5]], # Dummy features: year, mileage, engine size, has_accident, car_age
        [25000]  # Dummy target price
    )

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from form
        brand = request.form.get('brand')
        model_year = int(request.form.get('year'))
        mileage = float(request.form.get('mileage').replace(',', ''))
        fuel_type = request.form.get('fuel_type')
        engine_size = float(request.form.get('engine_size'))
        transmission = request.form.get('transmission')
        ext_color = request.form.get('ext_color')
        has_accident = int(request.form.get('has_accident'))
        
        # Calculate car age
        car_age = 2025 - model_year
        
        # Create DataFrame with input data
        input_data = pd.DataFrame({
            'brand': [brand],
            'model_year': [model_year],
            'mileage_cleaned': [mileage],
            'fuel_type': [fuel_type],
            'engine_size': [engine_size],
            'transmission_type': [transmission],
            'ext_col': [ext_color],
            'has_accident': [has_accident],
            'car_age': [car_age]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Format prediction as currency
        formatted_prediction = "${:,.2f}".format(prediction)
        
        return render_template('results.html', 
                              prediction=formatted_prediction,
                              brand=brand,
                              year=model_year,
                              mileage=mileage,
                              engine_size=engine_size,
                              color=ext_color,
                              transmission=transmission)
    
    except Exception as e:
        return render_template('index.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def predict_api():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Extract features
        brand = data.get('brand')
        model_year = int(data.get('year'))
        mileage = float(data.get('mileage'))
        fuel_type = data.get('fuel_type')
        engine_size = float(data.get('engine_size', 0))
        transmission = data.get('transmission')
        ext_color = data.get('ext_color')
        has_accident = int(data.get('has_accident'))
        
        # Calculate car age
        car_age = 2025 - model_year
        
        # Create DataFrame with input data
        input_data = pd.DataFrame({
            'brand': [brand],
            'model_year': [model_year],
            'mileage_cleaned': [mileage],
            'fuel_type': [fuel_type],
            'engine_size': [engine_size],
            'transmission_type': [transmission],
            'ext_col': [ext_color],
            'has_accident': [has_accident],
            'car_age': [car_age]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        return jsonify({
            'prediction': float(prediction),
            'formatted_prediction': "${:,.2f}".format(prediction)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
