from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
import uvicorn
import pandas as pd
import mlflow
import json
import joblib
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
import logging
import os
from sklearn import set_config
from scripts.data_clean_utils import perform_data_cleaning

# set the output as pandas
set_config(transform_output='pandas')

# initialize dagshub
import dagshub
import mlflow.client

dagshub.init(repo_owner='AmitZala', repo_name='swiggy-delivery-time-prediction', mlflow=True)

# set the mlflow tracking server
mlflow.set_tracking_uri("https://dagshub.com/AmitZala/swiggy-delivery-time-prediction.mlflow")

class Data(BaseModel):  
    ID: str
    Delivery_person_ID: str
    Delivery_person_Age: str
    Delivery_person_Ratings: str
    Restaurant_latitude: float
    Restaurant_longitude: float
    Delivery_location_latitude: float
    Delivery_location_longitude: float
    Order_Date: str
    Time_Orderd: str
    Time_Order_picked: str
    Weatherconditions: str
    Road_traffic_density: str
    Vehicle_condition: int
    Type_of_order: str
    Type_of_vehicle: str
    multiple_deliveries: str
    Festival: str
    City: str

    
    
def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
        
    return run_info


def load_transformer(transformer_path):
    transformer = joblib.load(transformer_path)
    return transformer



# columns to preprocess in data
num_cols = ["age",
            "ratings",
            "pickup_time_minutes",
            "distance"]

nominal_cat_cols = ['weather',
                    'type_of_order',
                    'type_of_vehicle',
                    "festival",
                    "city_type",
                    "is_weekend",
                    "order_time_of_day"]

ordinal_cat_cols = ["traffic","distance_type"]

#mlflow client
client = MlflowClient()

# load the model info to get the model name
model_name = load_model_information("run_information.json")['model_name']

# Decide whether to attempt loading from MLflow registry. By default we
# load the local artifact to avoid MLflow registry calls that may fail on
# hosting providers that don't support the full Model Registry API
# (for example DagsHub). Set environment variable `USE_MLFLOW_REGISTRY=1`
# to opt into registry loading.
use_registry = os.getenv("USE_MLFLOW_REGISTRY", "false").lower() in ("1", "true", "yes")

local_model_path = "models/model.joblib"

if use_registry:
    # stage of the model
    stage = "Production"
    model_path = f"models:/{model_name}/{stage}"
    try:
        model = mlflow.sklearn.load_model(model_path)
    except MlflowException as exc:
        logging.getLogger("app").warning(
            f"Could not load model from registry ({model_path}): {exc}. Falling back to local model file.")
        model = joblib.load(local_model_path)
else:
    # Load the local model artifact directly (default behaviour)
    logging.getLogger("app").info("Loading model from local artifact: %s", local_model_path)
    model = joblib.load(local_model_path)

# load the preprocessor
preprocessor_path = "models/preprocessor.joblib"
preprocessor = load_transformer(preprocessor_path)

# build the model pipeline
model_pipe = Pipeline(steps=[
    ('preprocess',preprocessor),
    ("regressor",model)
])

# create the app
app = FastAPI()

# create the home endpoint
@app.get(path="/")
def home():
    return "Welcome to the Swiggy Food Delivery Time Prediction App"

# create the predict endpoint
@app.post(path="/predict")
def do_predictions(data: Data):
    pred_data = pd.DataFrame({
        'ID': data.ID,
        'Delivery_person_ID': data.Delivery_person_ID,
        'Delivery_person_Age': data.Delivery_person_Age,
        'Delivery_person_Ratings': data.Delivery_person_Ratings,
        'Restaurant_latitude': data.Restaurant_latitude,
        'Restaurant_longitude': data.Restaurant_longitude,
        'Delivery_location_latitude': data.Delivery_location_latitude,
        'Delivery_location_longitude': data.Delivery_location_longitude,
        'Order_Date': data.Order_Date,
        'Time_Orderd': data.Time_Orderd,
        'Time_Order_picked': data.Time_Order_picked,
        'Weatherconditions': data.Weatherconditions,
        'Road_traffic_density': data.Road_traffic_density,
        'Vehicle_condition': data.Vehicle_condition,
        'Type_of_order': data.Type_of_order,
        'Type_of_vehicle': data.Type_of_vehicle,
        'multiple_deliveries': data.multiple_deliveries,
        'Festival': data.Festival,
        'City': data.City
        },index=[0]
    )
    # clean the raw input data
    cleaned_data = perform_data_cleaning(pred_data)
    # get the predictions
    predictions = model_pipe.predict(cleaned_data)[0]

    return predictions
   
   
if __name__ == "__main__":
    # Bind to localhost so the app is reachable at http://127.0.0.1:8000
    uvicorn.run(app="app:app", host="127.0.0.1", port=8000)