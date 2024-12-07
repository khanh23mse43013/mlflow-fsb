import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from itertools import product
import joblib

# Import models
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Create dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models and parameter grids
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42),
}

param_grids = {
    "RandomForest": {
        "n_estimators": [10, 50, 100],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5, 10],
    },
    "SVM": {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
    },
}

# Set up MLflow tracking
mlflow.set_tracking_uri("http://127.0.0.1:5000")
# mlflow.set_experiment("Model Training Experiment")

# Initialize variables for the best model
best_model = None
best_accuracy = 0
best_model_name = ""

# Training loop
for model_name, model in models.items():
    print(f"Training model: {model_name}")
    param_grid = param_grids[model_name]
    param_combinations = product(*param_grid.values())

    for params in param_combinations:
        param_dict = dict(zip(param_grid.keys(), params))
        print(f"Training with parameters: {param_dict}")

        with mlflow.start_run():
            # Configure model with parameters
            model.set_params(**param_dict)

            # Train the model
            model.fit(X_train, y_train)

            # Evaluate the model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")

            # Log parameters and metrics
            mlflow.log_param("model_name", model_name)
            mlflow.log_params(param_dict)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)
            # Log the model
            mlflow.sklearn.log_model(model, artifact_path="model")
            # Update the best model if needed
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_model_name = model_name
            # compare model in mlflow ui
            
            


# Save the best model locally
joblib.dump(best_model, "best_model.pkl")
# Save the best model to MLflow
mlflow.sklearn.log_model(best_model, artifact_path="best_model")
print(f"Best model: {best_model_name} with accuracy: {best_accuracy}")
