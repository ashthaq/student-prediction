import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import(
    accuracy_score,
    confusion_matrix,
    classification_report
)
# Step 1 Set project paths
BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_PATH = BASE_DIR/"dataset/student_pass_fail_classification.csv"
MODEL_PATH = BASE_DIR/"ml"/"model.pkl"

# step 2 load csv
df = pd.read_csv(DATASET_PATH)
# step 3 select feature and target
X = df[["StudyHoursPerDay","AttendanceRate"]]
y =df["Result"]

# step 4 split dataset
X_train ,X_test , y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
# Step 5 create ML Pipeline
Pipeline = Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler()),
    ("model",LogisticRegression(class_weight="balanced"))
])
# step 6 train model
Pipeline.fit(X_train,y_train)
print("Model training Completed!!")

# step 7 make a prediction using threshold 0.6 
fail_index = list(Pipeline.classes_).index("Fail")
fail_probability = Pipeline.predict_proba(X_test)[:,fail_index]
threshold = 0.6

y_pred = [
    "Fail" if probability >= threshold else "Pass"
    for probability in fail_probability
]
# step 8 Evaluate model 
print("\nAccuracy :",accuracy_score(y_test,y_pred))
print("\nConfusion matrix:",)
print(confusion_matrix(
    y_test,
    y_pred,
    labels=["Fail","Pass"]
))
print("\nClassification report:")
print(classification_report(y_test,y_pred))
# step 9 save trained Pipeline
joblib.dump(
    {
        "pipeline":Pipeline,
        "threshold":threshold
    },
    MODEL_PATH
)
print("\nModel saved Successfully!!!")
print("Model Location :",MODEL_PATH)
