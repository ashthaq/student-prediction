from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import pandas as pd
import joblib

from pathlib import Path

# Step 1 Load saved ML model 
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR/"ml"/"model.pkl"
saved_data = joblib.load(MODEL_PATH)

pipeline = saved_data["pipeline"]
threshold = saved_data["threshold"]

# Step 2 creat FastAPI Application
app = FastAPI(
    title="Student Pass Fail Prediction API",
    description="API for predicting student results",
    version="1.0"
)
# step 3 allow frontend communication 
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:5501",
        "http://localhost:5501"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# step 4 define input data
class StudentData(BaseModel):
    StudyHoursPerDay:float=Field(ge=0,le=24)
    AttendanceRate:float = Field(ge=0,le=100)

# step 5 Home route
@app.get("/")
def home():
    return{
        "message":"Student prediction API is running!"
    }

# step 6 prediction API
@app.post("/predict")
def predict(data:StudentData):
    input_data = pd.DataFrame([{
        "StudyHoursPerDay":data.StudyHoursPerDay,
        "AttendanceRate" : data.AttendanceRate
    }])
    # Get Fail Probability

    fail_index = list(pipeline.classes_).index("Fail")
    fail_probability = pipeline.predict_proba(
        input_data
    )[0][fail_index]

    # Apply threshold
    prediction = (
        "Fail"
        if fail_probability >= threshold
        else "Pass"
    )
    return {
        "prediction":prediction,
        "fail_probability":round(float(fail_probability),4),
        "threshold":threshold
    }
