import mlflow
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os

# Load Iris dataset
iris = load_iris()

# Split dataset into X features and Target variable
X = pd.DataFrame(data = iris["data"], columns= iris["feature_names"])
y = pd.Series(data = iris["target"], name="target")

# Split our training set and our test set 
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Visualize dataset 
X_train.head()

# Set your Hugging Face MLflow URL
os.environ["APP_URI"] = "https://msintech-mlflow-test.hf.space"

# Set your MLflow experiment name
EXPERIMENT_NAME = "my-first-mlflow-experiment"

# Set tracking URI to your Hugging Face MLflow
mlflow.set_tracking_uri(os.environ["APP_URI"])

# Set experiment's info 
mlflow.set_experiment(EXPERIMENT_NAME)

# Get our experiment info
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

# Call mlflow autolog
mlflow.sklearn.autolog()

# Start MLflow run
with mlflow.start_run(experiment_id = experiment.experiment_id):
    # Specified Parameter 
    c = 0.5

    # Instantiate and fit the model 
    lr = LogisticRegression(C=c)
    lr.fit(X_train.values, y_train.values)

    # Store metrics 
    predicted_qualities = lr.predict(X_test.values)
    accuracy = lr.score(X_test.values, y_test.values)

    # Print results 
    print("LogisticRegression model")
    print(f"Accuracy: {accuracy}")

    # Log Param
    mlflow.log_param("C", c)

    # Log Metric 
    mlflow.log_metric("Accuracy", accuracy)

    # Log model 
    mlflow.sklearn.log_model(lr, "model")