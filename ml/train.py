import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
# step 1 load csv
df = pd.read_csv("dataset/student_pass_fail_classification.csv")
# Step 2 understand data
print(df.head())
print(df.info())
# step 3 Check missing values .
print("\nMissing values")
print(df.isnull().sum())
# step 4 Check duplicates 
print("\nDuplicates")
print(df.duplicated().sum())
# step 5 Check target distribution 
print("\nTarget distibution :")
print(df["Result"].value_counts())
# step 6 Select features and targets 
X = df[["StudyHoursPerDay","AttendanceRate"]]
y = df["Result"]
# step 7 train_test_split
X_train , X_test , y_train , y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("\nTraining data :",X_train.shape)
print("\nTesting data :",X_test.shape)

# Step 8 handle missing values 
imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)
# step 9 feature scaling 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nPreprocessing completed !!")
print("\nTraining Shape :",X_train.shape)
print("\nTesting shape :",X_test.shape)

# --------- completed Preprocessing -------------------------

# PHASE 2: Machine Learning Model Training.

# step 10 create logistic regression
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
# train the model 
model.fit(X_train,y_train)
print("Model Training Completed !!")
# ------------- Model training Completed ------------
# Step 11 Make a Prediction 
y_pred = model.predict(X_test)
print("\nPredictions:")
print(y_pred)

# Step 12 Model evaluation 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# accuracy
accuracy = accuracy_score(y_test,y_pred)

print("\nModel accuary :",accuracy * 100,"%")

# confusion_matrix 
print("\nconfusion matrix:")
print(confusion_matrix(y_test,y_pred))
# classification_report
print("\nClassification report :")
print(classification_report(y_test,y_pred))

# ----------------- Compare ML algorithm --------------

# Step 13 Compare ML model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report

models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced"
    ),
    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42
    ),
    "Random Forest ":RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_estimators=100
    )
}
for name , model in models.items():
    # Train model
    model.fit(X_train,y_train)
    # Make predictions
    y_pred = model.predict(X_test)
    # Evaluate 
    print("\n===========================================")
    print(name)
    print("\n============================================")

    print("Accuracy:",accuracy_score(y_test,y_pred))
    print("\nClassification report :")
    print(classification_report(y_test,y_pred))
    # ---------------------------------------------------------------------------

# step 14 Experiment with classification thresholds

# create balanced LogisticRegression
balanced_lr = LogisticRegression(class_weight="balanced")
#train model 
balanced_lr.fit(X_train,y_train)

# Get probability of Fail.
fail_index =list(balanced_lr.classes_).index("Fail")
fail_probability = balanced_lr.predict_proba(X_test)[:,fail_index]

# Test different thresholds
thresholds = [0.3,0.4,0.5,0.6]

for threshold in thresholds:
    y_pred_treshold = [
        "Fail" if probability >= threshold else "Pass"
        for probability in fail_probability
    ]
    print("\n================================================")
    print("Threshold :",threshold)
    print("\n================================================")
    print("Accuracy:",accuracy_score(y_test,y_pred_treshold))
    print("\nConfusion matrix:")
    print(confusion_matrix(
        y_test,
        y_pred_treshold,
        labels=["Fail","Pass"]
    ))
    print("\nclassification Report:")
    print(classification_report(
        y_test,
        y_pred_treshold
    ))




