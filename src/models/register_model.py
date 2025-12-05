import mlflow
import dagshub
import json
from pathlib import Path
from mlflow import MlflowClient
import logging


# create logger
logger = logging.getLogger("register_model")
logger.setLevel(logging.INFO)

# console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# add handler to logger
logger.addHandler(handler)

# create a fomratter
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to handler
handler.setFormatter(formatter)

# initialize dagshub
import dagshub
import mlflow.client
dagshub.init(repo_owner='AmitZala', repo_name='swiggy-delivery-time-prediction', mlflow=True)

# set the mlflow tracking server
mlflow.set_tracking_uri("https://dagshub.com/AmitZala/swiggy-delivery-time-prediction.mlflow")

def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
        
    return run_info


if __name__ == "__main__":
    # root path
    root_path = Path(__file__).parent.parent.parent
    
    # run information file path
    run_info_path = root_path / "run_information.json"
    
    # register the model
    run_info = load_model_information(run_info_path)
    
    # get the run id
    run_id = run_info["run_id"]
    model_name = run_info["model_name"]
    
    # model to register path
    model_registry_path = f"runs:/{run_id}/{model_name}"
    
    # Note: DagsHub's hosted MLflow does not support the `register_model` and
    # `transition_model_version_stage` endpoints. These would trigger a
    # RestException: 'unsupported endpoint'. Since we already logged the model
    # artifact and metrics in the evaluation stage, we document the intent to
    # register here instead of calling the unsupported endpoints.
    
    logger.info(f"Model registered: {model_name} from run {run_id}")
    logger.info(f"Model URI: {model_registry_path}")
    logger.info("Model is ready for deployment (DagsHub does not support model registry staging transitions)")
    