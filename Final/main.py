"""
CS 471 Final Project: Fraud Detection using Machine Learning
Team 2: Shane Dyrdahl, Alastair Raymond, Riley Saliba, Dax Taraleskof
Date: November 2024

Fraud detection using machine learning models.
We preprocess the Fraud dataset, address class imbalance with SMOTE, and evaluate
the performance of Decision Tree and Random Forest classifiers. The models are tuned
using Ray's distributed hyperparameter tuning with recall as the primary metric.
"""

from time import perf_counter
import pickle
import pandas as pd
import dask.dataframe as dd
import ray
from ray import tune
from ray.util.joblib import register_ray
from ray.air import session  # Import session from ray.air
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

cpu_per = {"cpu": 4}
max_trials = 4

# Initialize Ray
ray.init(
    num_cpus=20,
    include_dashboard=False,
    object_store_memory=2 * 1024**3  # Limit to 2GB
)

register_ray()

# Dataset Path
dataset_path = "C:/pythonProject/AI/Final/dataset"

# ---- STEP 1: LOAD DATASET ----
start_time_csv = perf_counter()

try:
    with open(dataset_path, "rb") as file:
        dataset = pickle.load(file)
except FileNotFoundError:
    # Dask loads CSV in partitions
    ddf = dd.read_csv("Fraud.csv", blocksize="64MB")
    ddf = ddf.categorize(columns=["type", "nameOrig", "nameDest"])
    dataset = ddf.compute()
    with open(dataset_path, "wb") as fp:
        pickle.dump(dataset, fp)

end_time_csv = perf_counter()
print(f"-----> CSV Reading Time: {end_time_csv - start_time_csv:.4f} seconds")

# ---- STEP 2: ENCODE CATEGORICAL VARIABLES ----
start_time_encoding = perf_counter()

@ray.remote
def encode_column(column):
    """Encode a single column using Pandas' Categorical encoding."""
    return pd.Categorical(column).codes

categorical_columns = ["type", "nameOrig", "nameDest"]
encoded_columns = ray.get([encode_column.remote(dataset[col]) for col in categorical_columns])

for col, encoded_col in zip(categorical_columns, encoded_columns):
    dataset[col] = encoded_col

end_time_encoding = perf_counter()
print(f"-----> Encoding Time: {end_time_encoding - start_time_encoding:.4f} seconds")

# ---- STEP 3: SPLIT DATASET AND APPLY SMOTE ----
start_time_split_smote = perf_counter()

X = dataset[["type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest"]]
y = dataset["isFraud"]

# Split dataset into train/test
training_X, testing_X, training_y, testing_y = train_test_split(X, y, test_size=0.20)

# Apply SMOTE for oversampling
smote = SMOTE(random_state=130)
training_X, training_y = smote.fit_resample(training_X, training_y)

# Split training data into train/evaluation sets
training_X, evaluation_X, training_y, evaluation_y = train_test_split(training_X, training_y, test_size=0.20)

end_time_split_smote = perf_counter()
print(f"-----> Split & SMOTE Time: {end_time_split_smote - start_time_split_smote:.4f} seconds")

# ---- STEP 4: RULES-BASED EVALUATION ---
def evaluate_rules_based(threshold, data, actual):
    """
    Evaluate a rules-based model for fraud detection.

    Parameters:
        threshold (float): Transaction amount threshold for fraud detection.
        data (DataFrame): Dataset containing transaction features.
        actual (Series): True labels for fraud (1) or non-fraud (0).

    Outputs:
        Prints recall, precision, and F1-score for the rules-based model.
    """
    false_negative_count = 0
    true_positive_count = 0
    merged_data = data.copy()
    merged_data["isFraud"] = actual  # Insert true labels into the dataset

    # Identify potential fraudulent transactions based on the threshold
    potential_fraud = merged_data.loc[merged_data["amount"] < threshold]
    actual_fraud = potential_fraud.loc[potential_fraud["isFraud"] == 1]
    not_fraud = potential_fraud.loc[potential_fraud["isFraud"] == 0]

    # Calculate recall, precision, and F1-score
    true_positive_count = len(actual_fraud)
    total_fraud_count = len(merged_data.loc[merged_data["isFraud"] == 1])
    false_negative_count = total_fraud_count - true_positive_count
    false_positive_count = len(not_fraud)

    # Recall
    recall = true_positive_count / (true_positive_count + false_negative_count) if total_fraud_count > 0 else 0
    # Precision
    precision = true_positive_count / (true_positive_count + false_positive_count) if true_positive_count > 0 else 0
    # F1-score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Print metrics
    print(f"Rules-Based Evaluation with Threshold {threshold}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1-Score: {f1_score:.4f}")

start_time_evaluation = perf_counter()
# Rules-based evaluation on the evaluation set
evaluate_rules_based(threshold=10000000, data=evaluation_X, actual=evaluation_y)

end_time_evaluation = perf_counter()
print(f"-----> Rules-Based Evaluation Time: {end_time_evaluation - start_time_evaluation:.4f} seconds")


def evaluate_model(model, X, y):
    predictions = model.predict(X)
    print(f"Recall: {recall_score(y, predictions):.4f}")
    print(f"Classification Report:\n{classification_report(y, predictions)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y, predictions)}")
    
# ---- STEP 5: TRAIN DECISION TREE MODEL USING RAY TUNE ----

# Use tune.with_parameters to handle large datasets
tune_training_X = ray.put(training_X)
tune_training_y = ray.put(training_y)
tune_evaluation_X = ray.put(evaluation_X)
tune_evaluation_y = ray.put(evaluation_y)


def train_decision_tree(config):
    """Train and evaluate a Decision Tree with given hyperparameters."""
    # Retrieve data from Ray's object store
    training_X = ray.get(tune_training_X)
    training_y = ray.get(tune_training_y)
    evaluation_X = ray.get(tune_evaluation_X)
    evaluation_y = ray.get(tune_evaluation_y)

    # Train the Decision Tree
    model = DecisionTreeClassifier(**config, random_state=130, class_weight="balanced")
    model.fit(training_X, training_y)

    # Predict and calculate recall
    predictions = model.predict(evaluation_X)
    recall = recall_score(evaluation_y, predictions)

    # Report recall score
    session.report({"recall": recall})


# Define Decision Tree hyperparameters
# dt_param_space = {
#     "max_depth": tune.grid_search([4]),
#     "min_samples_split": tune.grid_search([2])
# }

dt_param_space = {
    "max_depth": tune.grid_search([3, 4, 5]),
    "min_samples_split": tune.grid_search([2, 3, 4])
}

start_time_dt = perf_counter()

dt_analysis = tune.run(
    tune.with_parameters(train_decision_tree),
    config=dt_param_space,
    metric="recall",
    mode="max",
    resources_per_trial=cpu_per,
    max_concurrent_trials=max_trials,
    trial_dirname_creator=lambda trial: f"{trial.trial_id}"  # Shortens the directory path
)

best_dt_config = dt_analysis.best_config
print(f"Best Decision Tree Config: {best_dt_config}")
# Evaluate Decision Tree
best_dt = DecisionTreeClassifier(**best_dt_config, random_state=130, class_weight="balanced")
best_dt.fit(training_X, training_y)
print("Decision Tree Performance:")
evaluate_model(best_dt, testing_X, testing_y)

end_time_dt = perf_counter()
print(f"-----> Decision Tree Training Time: {end_time_dt - start_time_dt:.4f} seconds")


# ---- STEP 6: TRAIN RANDOM FOREST MODEL USING RAY TUNE ----
def train_random_forest(config):
    """Train and evaluate a Random Forest with given hyperparameters."""
    # Retrieve data from Ray's object store
    training_X = ray.get(tune_training_X)
    training_y = ray.get(tune_training_y)
    evaluation_X = ray.get(tune_evaluation_X)
    evaluation_y = ray.get(tune_evaluation_y)

    # Train the Random Forest
    model = RandomForestClassifier(
        **config,
        random_state=130,
        class_weight="balanced",
        n_jobs=-1
    )
    model.fit(training_X, training_y)

    # Predict and calculate recall
    predictions = model.predict(evaluation_X)
    recall = recall_score(evaluation_y, predictions)

    # Report recall score
    session.report({"recall": recall})


# Define Random Forest hyperparameters
# rf_param_space = {
#     "n_estimators": tune.grid_search([25]),
#     "max_depth": tune.grid_search([2]),
#     "min_samples_leaf": tune.grid_search([1])
# }

rf_param_space = {
    "n_estimators": tune.grid_search([10, 25, 50]),
    "max_depth": tune.grid_search([2, 3, 4]),
    "min_samples_leaf": tune.grid_search([1, 2])
}

start_time_rf = perf_counter()

rf_analysis = tune.run(
    tune.with_parameters(train_random_forest),
    config=rf_param_space,
    metric="recall",
    mode="max",
    resources_per_trial=cpu_per,
    max_concurrent_trials=max_trials,
    trial_dirname_creator=lambda trial: f"{trial.trial_id}"
)

best_rf_config = rf_analysis.best_config
print(f"Best Random Forest Config: {best_rf_config}")
# Evaluate Random Forest
best_rf = RandomForestClassifier(**best_rf_config, random_state=130, class_weight="balanced", n_jobs=-1)
best_rf.fit(training_X, training_y)
print("Random Forest Performance:")
evaluate_model(best_rf, testing_X, testing_y)

end_time_rf = perf_counter()
print(f"-----> Random Forest Training Time: {end_time_rf - start_time_rf:.4f} seconds")


# Shutdown Ray
ray.shutdown()

print(f"| ALL ELAPSED TIMES |")
print(f"| CSV Reading Time: {end_time_csv - start_time_csv:.4f} seconds")
print(f"| Encoding Time: {end_time_encoding - start_time_encoding:.4f} seconds")
print(f"| Split & SMOTE Time: {end_time_split_smote - start_time_split_smote:.4f} seconds")
print(f"| Rules-Based Evaluation Time: {end_time_evaluation - start_time_evaluation:.4f} seconds")
print(f"| Decision Tree Training Time: {end_time_dt - start_time_dt:.4f} seconds")
print(f"| Random Forest Training Time: {end_time_rf - start_time_rf:.4f} seconds")
