import pandas as pd
import numpy as np
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score, make_scorer, recall_score
# for model serialization
import joblib
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Wellness_Tourism_Package_Prediction_MLOps_Experiment")

api = HfApi(token=os.getenv("HF_TOKEN"))

Xtrain_path = "hf://datasets/bkrishnamukund/Wellness-Tourism-Prediction/Xtrain.csv"
Xtest_path = "hf://datasets/bkrishnamukund/Wellness-Tourism-Prediction/Xtest.csv"
ytrain_path = "hf://datasets/bkrishnamukund/Wellness-Tourism-Prediction/ytrain.csv"
ytest_path = "hf://datasets/bkrishnamukund/Wellness-Tourism-Prediction/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

# One-hot encode 'Type' and scale numeric features
numeric_features = [
    "Age",
    "CityTier",
    "NumberOfPersonVisiting",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "Passport",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "MonthlyIncome",
    "PitchSatisfactionScore",
    "NumberOfFollowups",
    "DurationOfPitch"
]

categorical_features = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "MaritalStatus",
    "Designation",
    "ProductPitched"
]

# Set the clas weight to handle class imbalance
scale_pos_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
scale_pos_weight

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight,
                              random_state=42)

# Define hyperparameter grid
param_grid  = {
    "xgbclassifier__n_estimators": [75, 100, 150],
    "xgbclassifier__max_depth": [2, 3, 4],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__colsample_bytree": [0.5, 0.7],
    "xgbclassifier__reg_lambda": [0.5, 1.0],
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Define scorer to focus on positive class
# Use recall if your goal is capture as many buyers as possible, even if you get some false positives.
scorer = make_scorer(recall_score, pos_label=1)


# ----------------------------
# Start the main MLflow run
# ----------------------------
with mlflow.start_run(run_name="RandomizedSearchCV_Recall"):

    # ----------------------------
    # Hyperparameter tuning
    # ----------------------------
    rand_search = RandomizedSearchCV(
        estimator=model_pipeline,
        param_distributions=param_grid,
        n_iter=50,
        scoring=scorer,  # e.g., recall_scorer
        cv=5,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    rand_search.fit(Xtrain, ytrain)

    # ----------------------------
    # Log all parameter combinations as nested runs
    # ----------------------------
    results = rand_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_cv_recall", mean_score)
            mlflow.log_metric("std_cv_recall", std_score)

    # ----------------------------
    # Log best parameters in main run
    # ----------------------------
    mlflow.log_params(rand_search.best_params_)

    # ----------------------------
    # Store best estimator
    # ----------------------------
    best_model = rand_search.best_estimator_

    # Log the best model with input example
    input_example = Xtrain.head(5)
    mlflow.sklearn.log_model(
        best_model,
        name="xgb_pipeline_best",
        input_example=input_example
    )

    # ----------------------------
    # Predict probabilities and tune threshold for recall
    # ----------------------------
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    thresholds = np.arange(0.3, 0.7, 0.01)
    best_thresh = 0.5
    best_recall = 0

    for t in thresholds:
        y_pred_train = (y_pred_train_proba >= t).astype(int)
        recall = recall_score(ytrain, y_pred_train)
        if recall > best_recall:
            best_recall = recall
            best_thresh = t

    classification_threshold = best_thresh
    print("Optimal threshold based on train Recall:", classification_threshold)

    # Log threshold in MLflow
    mlflow.log_param(
        "classification_threshold_for_positive_class_ProdTaken",
        classification_threshold
    )

    # Apply threshold
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)
    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    # ----------------------------
    # Compute and log metrics
    # ----------------------------
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    # Save the model locally
    model_path = "best_wellness_tourism_prediction_model_v2.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "bkrishnamukund/Wellness-Tourism-Prediction"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj="best_wellness_tourism_prediction_model_v2.joblib",
        path_in_repo="best_wellness_tourism_prediction_model_v2.joblib",
        repo_id=repo_id,
        repo_type=repo_type
    )

# ----------------------------
# End the main MLflow run
# ----------------------------
mlflow.end_run()
