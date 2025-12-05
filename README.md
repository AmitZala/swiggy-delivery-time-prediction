# 🍔 Swiggy Delivery Time Prediction

A machine learning project that predicts food delivery times using a complete MLOps pipeline with DVC, MLflow, and FastAPI. This project demonstrates end-to-end ML workflow from data cleaning to model serving.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Pipeline](#project-pipeline)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Model Performance](#model-performance)

## 🎯 Overview

This project predicts the delivery time for food orders placed on Swiggy (Indian food delivery platform) based on features like:
- Delivery person details (age, ratings, experience)
- Restaurant and delivery location coordinates
- Weather conditions and traffic density
- Order type and vehicle type
- Time of day and festival information

**Key Technologies:**
- **Data Pipeline:** DVC (Data Version Control)
- **ML Workflow:** Scikit-learn, LightGBM, Random Forest
- **Model Tracking:** MLflow + DagsHub
- **API Server:** FastAPI + Uvicorn
- **Model Serialization:** Joblib

---

## 📊 Project Pipeline

The project follows a modular DVC pipeline with the following stages:

```
┌─────────────────┐
│  Raw Data       │
│  (swiggy.csv)   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│ Stage 1: Data Cleaning      │
│ - Handle missing values     │
│ - Remove outliers           │
│ - Feature engineering       │
└────────┬────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│ Stage 2: Data Preparation    │
│ - Train/Test split (75/25)   │
│ - Save split datasets        │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│ Stage 3: Data Preprocessing          │
│ - Numerical: MinMaxScaler            │
│ - Categorical: OneHotEncoder         │
│ - Ordinal: OrdinalEncoder            │
│ - Save preprocessor artifact         │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│ Stage 4: Model Training              │
│ - Random Forest Regressor            │
│ - LightGBM Regressor                 │
│ - Stacking Ensemble                  │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│ Stage 5: Model Evaluation            │
│ - Calculate MAE, R² scores           │
│ - Cross-validation                   │
│ - Log metrics to MLflow              │
└──────────────────────────────────────┘
```

**Run the entire pipeline:**
```bash
dvc repro
```

---

## 📁 Project Structure

```
swiggy-delivery-time-prediction/
│
├── app.py                          # FastAPI application for model serving
├── Dockerfile                       # Docker configuration for containerization
├── requirements.txt                 # Core dependencies
├── requirements-dev.txt             # Development dependencies
├── requirements-dockers.txt         # Docker-specific dependencies
├── .env.example                     # Environment variables template
├── .gitignore                       # Git ignore rules
├── .gitattributes                   # Git LFS configuration
├── dvc.yaml                         # DVC pipeline definition
├── dvc.lock                         # DVC lock file (tracked in Git)
├── params.yaml                      # Hyperparameter configuration
├── README.md                        # This file
│
├── data/
│   ├── raw/                         # Original dataset
│   │   └── swiggy.csv
│   ├── cleaned/                     # Cleaned data
│   │   └── swiggy_cleaned.csv
│   ├── interim/                     # Train/test split
│   │   ├── train.csv
│   │   └── test.csv
│   └── processed/                   # Preprocessed data
│       ├── train_trans.csv
│       └── test_trans.csv
│
├── models/                          # Trained model artifacts
│   ├── model.joblib                 # Stacking ensemble model
│   ├── preprocessor.joblib          # ColumnTransformer for preprocessing
│   ├── power_transformer.joblib     # Power transformer artifact
│   └── stacking_regressor.joblib    # Stacking regressor component
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_cleaning.py         # Data cleaning stage
│   │   ├── data_preparation.py      # Train/test split stage
│   │   └── data_preprocessing.py    # Feature preprocessing stage
│   ├── features/
│   │   ├── __init__.py
│   │   └── data_preprocessing.py    # Feature engineering (wrapper)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py                 # Model training stage
│   │   ├── evaluation.py            # Model evaluation stage
│   │   └── register_model.py        # Model registration (optional)
│   └── visualization/
│       └── __init__.py
│
├── scripts/
│   ├── data_clean_utils.py          # Data cleaning utilities
│   ├── sample_predictions.py        # Example prediction script
│   └── promote_model_to_prod.py     # Model promotion script (optional)
│
├── tests/
│   ├── test_model_registry.py       # Model registry tests
│   └── test_model_perf.py           # Model performance tests
│
├── notebooks/                       # Jupyter notebooks for exploration
│   └── (exploratory analysis)
│
├── reports/
│   └── figures/                     # Generated figures and reports
│
├── docs/                            # Documentation
│   ├── commands.rst
│   ├── getting-started.rst
│   └── index.rst
│
└── references/                      # Reference materials
```

---

## 🔧 Installation

### Prerequisites

- Python 3.11+
- pip or conda
- Docker (optional, for containerization)
- Git and Git LFS

### Step 1: Clone the Repository

```bash
git clone https://github.com/AmitZala/swiggy-delivery-time-prediction.git
cd swiggy-delivery-time-prediction
```

### Step 2: Create Virtual Environment

**Using venv:**
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

**Using conda:**
```bash
conda create -n swiggy-delivery python=3.11
conda activate swiggy-delivery
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install python-dotenv
```

### Step 4: Install DVC and Git LFS

```bash
pip install dvc
git lfs install
```

### Step 5: Pull DVC Data (Optional)

```bash
dvc pull
```

---

## ⚙️ Configuration

### Environment Variables

1. Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

2. Edit `.env` with your actual credentials:
```env
# DagsHub Configuration
DAGSHUB_REPO_OWNER=YourUsername
DAGSHUB_REPO_NAME=your-repo-name

# MLflow Configuration
MLFLOW_TRACKING_URI=https://dagshub.com/YourUsername/your-repo-name.mlflow
MLFLOW_EXPERIMENT_NAME=DVC Pipeline

# FastAPI Configuration
FASTAPI_HOST=127.0.0.1
FASTAPI_PORT=8000

# Model and Data Paths
MODEL_PATH=models/model.joblib
PREPROCESSOR_PATH=models/preprocessor.joblib

# Other configurations
USE_MLFLOW_REGISTRY=false
TARGET_COLUMN=time_taken
```

### DVC Remote Configuration (Optional)

```bash
dvc remote add -d dagshub "https://dagshub.com/YourUsername/your-repo-name.dvc"
dvc remote modify dagshub --local auth basic
dvc remote modify dagshub --local user YourUsername
dvc remote modify dagshub --local password "YOUR_DAGSHUB_TOKEN"
```

---

## 🚀 Usage

### 1. Run the Complete Pipeline

```bash
cd d:\swiggy-delivery-time-prediction
dvc repro
```

This will execute all stages:
- Data cleaning
- Data preparation (train/test split)
- Data preprocessing (feature transformation)
- Model training
- Model evaluation

### 2. Start the FastAPI Server

```bash
python app.py
```

The API will be available at: `http://127.0.0.1:8000`

### 3. Access Interactive API Documentation

Open your browser and go to:
- **Swagger UI:** `http://127.0.0.1:8000/docs`
- **ReDoc:** `http://127.0.0.1:8000/redoc`

### 4. Make Predictions via API

**Using Python requests:**
```python
import requests

payload = {
    "ID": "1",
    "Delivery_person_ID": "DP_001",
    "Delivery_person_Age": "28",
    "Delivery_person_Ratings": "4.5",
    "Restaurant_latitude": 12.9716,
    "Restaurant_longitude": 77.5946,
    "Delivery_location_latitude": 12.9352,
    "Delivery_location_longitude": 77.6245,
    "Order_Date": "2022-01-01",
    "Time_Orderd": "12:00",
    "Time_Order_picked": "12:10",
    "Weatherconditions": "Sunny",
    "Road_traffic_density": "Medium",
    "Vehicle_condition": 5,
    "Type_of_order": "Food",
    "Type_of_vehicle": "Bike",
    "multiple_deliveries": "1",
    "Festival": "No",
    "City": "Bengaluru"
}

response = requests.post("http://127.0.0.1:8000/predict", json=payload)
print(f"Predicted delivery time: {response.json()} minutes")
```

**Using curl:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "ID":"1",
    "Delivery_person_ID":"DP_001",
    "Delivery_person_Age":"28",
    "Delivery_person_Ratings":"4.5",
    "Restaurant_latitude":12.9716,
    "Restaurant_longitude":77.5946,
    "Delivery_location_latitude":12.9352,
    "Delivery_location_longitude":77.6245,
    "Order_Date":"2022-01-01",
    "Time_Orderd":"12:00",
    "Time_Order_picked":"12:10",
    "Weatherconditions":"Sunny",
    "Road_traffic_density":"Medium",
    "Vehicle_condition":5,
    "Type_of_order":"Food",
    "Type_of_vehicle":"Bike",
    "multiple_deliveries":"1",
    "Festival":"No",
    "City":"Bengaluru"
  }'
```

### 5. Run Sample Predictions

```bash
python scripts/sample_predictions.py
```

---

## 🐳 Docker Usage

### Build Docker Image

```bash
docker build -t amitzala93/swiggy-delivery-time-prediction:latest .
```

### Run Container Locally

```bash
docker run -p 8000:8000 amitzala93/swiggy-delivery-time-prediction:latest
```

### Push to Docker Hub

```bash
docker login
docker push amitzala93/swiggy-delivery-time-prediction:latest
```

---

## 📡 API Documentation

### Endpoints

#### 1. **Home Endpoint**
- **Route:** `GET /`
- **Description:** Welcome message
- **Response:** `"Welcome to the Swiggy Food Delivery Time Prediction App"`

#### 2. **Predict Endpoint**
- **Route:** `POST /predict`
- **Description:** Make delivery time predictions
- **Input:** JSON with delivery details (see Usage section)
- **Output:** Predicted delivery time in minutes (float)

#### 3. **API Documentation**
- **Swagger UI:** `GET /docs`
- **ReDoc:** `GET /redoc`
- **OpenAPI Schema:** `GET /openapi.json`

---

## 📈 Model Performance

The trained ensemble model combines:
- **Random Forest Regressor** (479 estimators, max_depth=17)
- **LightGBM Regressor** (154 estimators, max_depth=27)

**Hyperparameters are defined in `params.yaml`**

### Evaluation Metrics

Metrics are logged to MLflow and include:
- **Mean Absolute Error (MAE)** - Training & Testing
- **R² Score** - Training & Testing
- **Cross-Validation Scores** (5-fold CV)

---

## 🔄 DVC Pipeline Stages

### Stage 1: data_cleaning
```bash
cmd: python src/data/data_cleaning.py
deps:
  - data/raw/swiggy.csv
  - src/data/data_cleaning.py
outs:
  - data/cleaned/swiggy_cleaned.csv
```

### Stage 2: data_preparation
```bash
cmd: python src/data/data_preparation.py
params:
  - Data_Preparation.test_size
  - Data_Preparation.random_state
deps:
  - data/cleaned/swiggy_cleaned.csv
  - src/data/data_preparation.py
outs:
  - data/interim/train.csv
  - data/interim/test.csv
```

### Stage 3: data_preprocessing
```bash
cmd: python src/features/data_preprocessing.py
deps:
  - data/interim/train.csv
  - data/interim/test.csv
  - src/features/data_preprocessing.py
outs:
  - data/processed/train_trans.csv
  - data/processed/test_trans.csv
  - models/preprocessor.joblib
```

### Stage 4: train
```bash
cmd: python src/models/train.py
params:
  - Train.Random_Forest
  - Train.LightGBM
deps:
  - src/models/train.py
  - data/processed/train_trans.csv
outs:
  - models/model.joblib
  - models/power_transformer.joblib
  - models/stacking_regressor.joblib
```

### Stage 5: evaluation
```bash
cmd: python src/models/evaluation.py
deps:
  - src/models/evaluation.py
  - data/processed/train_trans.csv
  - data/processed/test_trans.csv
  - models/model.joblib
outs:
  - run_information.json
```

---

## 🛠️ Development

### Run Tests

```bash
pytest tests/ -v
```

### Check Code Quality

```bash
pylint src/ --disable=all --enable=E
```

### View DVC Status

```bash
dvc status -c
dvc dag
```

### Push DVC Cache

```bash
dvc push
```

---

## 📝 Environment Variables Reference

| Variable | Example | Purpose |
|----------|---------|---------|
| `DAGSHUB_REPO_OWNER` | `AmitZala` | DagsHub repository owner |
| `DAGSHUB_REPO_NAME` | `swiggy-delivery-time-prediction` | DagsHub repository name |
| `MLFLOW_TRACKING_URI` | `https://dagshub.com/...mlflow` | MLflow server URI |
| `MLFLOW_EXPERIMENT_NAME` | `DVC Pipeline` | MLflow experiment name |
| `FASTAPI_HOST` | `127.0.0.1` | FastAPI server host |
| `FASTAPI_PORT` | `8000` | FastAPI server port |
| `MODEL_PATH` | `models/model.joblib` | Path to trained model |
| `USE_MLFLOW_REGISTRY` | `false` | Enable MLflow model registry |

---

## 🐛 Troubleshooting

### Issue: `dvc pull` fails with missing cache files
**Solution:** Ensure DVC remote is configured and has sufficient permissions.

### Issue: `pylint: command not found` in CI
**Solution:** Install pylint: `pip install pylint`

### Issue: FastAPI app won't start
**Solution:** Check if port 8000 is already in use or verify `.env` configuration.

### Issue: Model not loading from MLflow registry
**Solution:** Set `USE_MLFLOW_REGISTRY=false` in `.env` to load from local storage.

---

## 📚 Resources

- [DVC Documentation](https://dvc.org/doc)
- [MLflow Documentation](https://mlflow.org/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [Scikit-learn Documentation](https://scikit-learn.org)
- [LightGBM Documentation](https://lightgbm.readthedocs.io)

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👤 Author

**Amit Zala**
- GitHub: [@AmitZala](https://github.com/AmitZala)
- Project: [swiggy-delivery-time-prediction](https://github.com/AmitZala/swiggy-delivery-time-prediction)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📧 Contact

For questions or suggestions, please open an issue on GitHub or contact the maintainer.

---

**Last Updated:** December 2025
