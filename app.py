import sys
sys.path.append('/home/western/ds_projects/electricity-bills')

from flask import Flask, request, render_template
import os
import pandas as pd
import sys
import numpy as np

from src.ElectricityBill.pipelines.pip_07_prediction_pipeline import *
from src.ElectricityBill.exception import CustomException
from src.ElectricityBill import logger

# Create flask app
app = Flask(__name__)

# Configure debug mode
if os.getenv('DEBUG') == True:
    app.debug = True


# Custom data class for data preprocessing
class CustomData:
    def __init__(self,
                 Fan: str, Refrigerator: float, AirConditioner: float, Television: float,
                 Month: str, MonthlyHours: str, TariffRate: str, TotalApplianceHours: str,
                 City: str, Company: str):

        # Corrected attributes names to match form fields

        self.Fan = Fan
        self.Refrigerator = Refrigerator
        self.AirConditioner = AirConditioner
        self.Television = Television
        self.Month = Month
        self.MonthlyHours = MonthlyHours
        self.TariffRate = TariffRate
        self.TotalApplianceHours = TotalApplianceHours
        self.City = City
        self.Company = Company

    def get_season(self, month: int) -> str:
        """Determines the season based on the month."""
        if month in [3, 4, 5]:
            return "Summer"
        elif month in [6, 7, 8, 9]:
            return "Monsoon"
        else:
            return "Winter"

    def get_usage_category(self, monthly_hours: float) -> str:
        """Determines the usage category based on monthly hours."""
        if 650 <= monthly_hours <= 1000:
            return "High"
        elif 400 <= monthly_hours <= 649:
            return "Medium"
        else:
            return "Low"

    def get_data_as_df(self) -> pd.DataFrame:
        try:
            # Transform input to dataframe
            month = int(self.Month)  # Convert Month to integer
            monthly_hours = float(self.MonthlyHours) #Convert MonthlyHours to float

            season = self.get_season(month)
            usage_category = self.get_usage_category(monthly_hours)

            data = {
                'Fan': [self.Fan],
                'Refrigerator': [self.Refrigerator],
                'AirConditioner': [self.AirConditioner],
                'Television': [self.Television],
                'Month': [month],
                'MonthlyHours': [monthly_hours],
                'TariffRate': [self.TariffRate],
                'TotalApplianceHours': [self.TotalApplianceHours],
                'City': [self.City],
                'Company': [self.Company],
                'season': [season],  # Include calculated Season
                'UsageCategory': [usage_category]  # Include calculated UsageCategory
            }
            df = pd.DataFrame(data)

            return df

        except Exception as e:
            logger.error("Error converting data to dataframe: {}".format(str(e)))
            raise CustomException(e, sys)

# Route to homepage
@app.route('/')  # Added the route decorator to the homepage function
def home():
    return render_template('index.html')

# Route to prediction
@app.route('/predict', methods=['GET', 'POST'])
def predict_date_point():
    if request.method == 'GET':
        return render_template('index.html')

    else:
        try:
            # Collect form data
            form_data = {
                'Fan': request.form['Fan'],
                'Refrigerator': request.form['Refrigerator'],
                'AirConditioner': request.form['AirConditioner'],
                'Television': request.form['Television'],
                'Month': request.form['Month'],
                'MonthlyHours': request.form['MonthlyHours'],
                'TariffRate': request.form['TariffRate'],
                'TotalApplianceHours': request.form['TotalApplianceHours'],
                'City': request.form['City'],
                'Company': request.form['Company'],
            }

            # Create CustomData object
            custom_data = CustomData(**form_data)

            # Transform to dataframe
            pred_df = custom_data.get_data_as_df()

            print("Columns in pred_df:", pred_df.columns)  # Add this line

            # Load the configuration and create a prediction pipeline
            config_manager = ConfigurationManager()
            prediction_config = config_manager.get_prediction_pipeline_config()
            prediction_pipeline = PredictionPipeline(prediction_config)

            # Perform prediction
            prediction = prediction_pipeline.make_predictions(pred_df)

            #logger.info(f"Prediction: {prediction}")


            # Return prediction
            return render_template('prediction.html', prediction=prediction[0])

        except Exception as e:
            # Log the full exception for detailed debugging
            logger.exception(f"Model prediction failed: {str(e)}")
            return render_template('home.html', error_message=f"Model prediction failed. {str(e)}. Please provide correct values.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081)