# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize the Flask application
sales_predictor_api = Flask("SuperKart Sales Prediction API")

# Load the trained machine learning model
model = joblib.load("/content/drive/MyDrive/AI&ML/Model Deployment/Week3 project/deployment_files/Superkart_forecast_sales_revenue_prediction_model_v1_0.joblib")

# Define a route for the home page (GET request)
@sales_predictor_api.get('/')
def home():
    return "Welcome to the SuperKart Sales Prediction API!"

# Define an endpoint for single property prediction (POST request)
@sales_predictor_api.post('/v1/predict')
def predict_sales():

    # Extract relevant features from the JSON data
    sample = {
    "Product_Weight": property_data["Product_Weight"],
    "Product_Sugar_Content": property_data["Product_Sugar_Content"],
    "Product_Allocated_Area": property_data["Product_Allocated_Area"],
    "Product_MRP": property_data["Product_MRP"],
    "Store_Size": property_data["Store_Size"],
    "Store_Location_City_Type": property_data["Store_Location_City_Type"],
    "Store_Type": property_data["Store_Type"],
    "Product_Prefix": property_data["Product_Id_char"],
    "Store_Age": property_data["Store_Age_Years"],
    "Product_Type_Category": property_data["Product_Type_Category"]
    }
    # Convert the extracted data into a Pandas DataFrame
    input_data = pd.DataFrame([sample])

    # Predict sales
    predicted_sales = model.predict(input_data)[0]

    # Convert NumPy datatype to Python float
    predicted_sales = round(float(predicted_sales), 2)

    # Return the actual price
    return jsonify({"Predicted Sales": predicted_sales})


# Define an endpoint for batch prediction (POST request)
@sales_predictor_api.post('/v1/predictbatch')
def predict_sales_batch():
    """
    This function handles POST requests to the '/v1/predictbatch' endpoint
    It expects a CSV file containing property details for multiple properties
    and returns the predicted sale as a dictionary in the JSON response.
    """
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the CSV file into a Pandas DataFrame
    input_data = pd.read_csv(file)

    # Predict
    predictions = model.predict(input_data)

    predictions = [round(float(x), 2) for x in predictions]

    # If Product_Id exists, use it as the key
    if 'Product_Id' in input_data.columns:
        ids = input_data['Product_Id'].tolist()
    else:
        ids = list(range(1, len(predictions) + 1))

    output = dict(zip(ids, predictions))

    return jsonify(output)

# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    sales_predictor_api.run(debug=True)
