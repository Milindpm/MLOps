
"""
Model Training Script
Author : Milind Mundankar
"""

import os
import joblib
import pandas as pd
import mlflow

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# ----------------------------------------------------------
# Load Train and Test Data
# ----------------------------------------------------------

TRAIN_PATH = "tourism_project/model_building/data/train.csv"
TEST_PATH = "tourism_project/model_building/data/test.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Train Shape :", train_df.shape)
print("Test Shape  :", test_df.shape)

# ----------------------------------------------------------
# Separate Features and Target
# ----------------------------------------------------------

TARGET = "ProdTaken"

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

# ----------------------------------------------------------
# Numerical and Categorical Columns
# ----------------------------------------------------------

categorical_cols = X_train.select_dtypes(include=["object"]).columns

numerical_cols = X_train.select_dtypes(exclude=["object"]).columns

# ----------------------------------------------------------
# Preprocessor
# ----------------------------------------------------------

preprocessor = make_column_transformer(

    (StandardScaler(), numerical_cols),

    (OneHotEncoder(handle_unknown="ignore"), categorical_cols)

)

# ----------------------------------------------------------
# Pipeline
# ----------------------------------------------------------

pipeline = make_pipeline(

    preprocessor,

    RandomForestClassifier(random_state=42)

)

# ----------------------------------------------------------
# Hyperparameter Grid
# ----------------------------------------------------------

param_grid = {

    "randomforestclassifier__n_estimators":[100,200],

    "randomforestclassifier__max_depth":[5,10,None],

    "randomforestclassifier__min_samples_split":[2,5]

}

# ----------------------------------------------------------
# Grid Search
# ----------------------------------------------------------

grid_search = GridSearchCV(

    pipeline,

    param_grid,

    cv=5,

    scoring="accuracy",

    n_jobs=-1

)

# ----------------------------------------------------------
# MLflow Tracking
# ----------------------------------------------------------

with mlflow.start_run(run_name="RandomForest_GridSearch"):

    # Train Model
    grid_search.fit(X_train, y_train)

    # ------------------------------------------------------
    # Log Every Parameter Combination
    # ------------------------------------------------------

    results = grid_search.cv_results_

    for i in range(len(results["params"])):

        with mlflow.start_run(

            nested=True,

            run_name=f"Run_{i+1}"

        ):

            params = results["params"][i]

            mean_score = results["mean_test_score"][i]

            std_score = results["std_test_score"][i]

            mlflow.log_params(params)

            mlflow.log_metric(

                "mean_test_score",

                mean_score

            )

            mlflow.log_metric(

                "std_test_score",

                std_score

            )

    # ------------------------------------------------------
    # Best Model
    # ------------------------------------------------------

    best_model = grid_search.best_estimator_

    predictions = best_model.predict(X_test)

    accuracy = accuracy_score(

        y_test,

        predictions

    )

    precision = precision_score(

        y_test,

        predictions

    )

    recall = recall_score(

        y_test,

        predictions

    )

    f1 = f1_score(

        y_test,

        predictions

    )

    # ------------------------------------------------------
    # Log Best Parameters
    # ------------------------------------------------------

    mlflow.log_params(

        grid_search.best_params_

    )

    # ------------------------------------------------------
    # Log Final Metrics
    # ------------------------------------------------------

    mlflow.log_metric(

        "accuracy",

        accuracy

    )

    mlflow.log_metric(

        "precision",

        precision

    )

    mlflow.log_metric(

        "recall",

        recall

    )

    mlflow.log_metric(

        "f1_score",

        f1

    )

    print()

    print("="*60)

    print("Classification Report")

    print("="*60)

    print(

        classification_report(

            y_test,

            predictions

        )

    )

# ----------------------------------------------------------
# Save Model
# ----------------------------------------------------------

MODEL_PATH = "tourism_project/deployment/best_model.pkl"

joblib.dump(best_model,MODEL_PATH)

print()

print("Best Model Saved Successfully")

print(MODEL_PATH)
