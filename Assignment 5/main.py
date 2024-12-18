# CS 471 AI - Assignment 5
# Perform classification using two machine learning algorithm
# By: Dax Taraleskof and Shane Dyrdahl
# 10/03/2024

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

# Load data
training_data = pd.read_csv("training.csv", header=None, usecols=[19, 23], names=['Time', 'Current'])
test_data = pd.read_csv("test.csv", header=None, usecols=[0, 4], names=['Time', 'Current'])

training_data = training_data[training_data['Time'] <= 5.4]
test_data = test_data[test_data['Time'] <= 2.4]

# Preprocessing
imputer = SimpleImputer(strategy='median')
training_data_preprocessed = pd.DataFrame(imputer.fit_transform(training_data[['Current']]), columns=['Current'])
test_data_preprocessed = pd.DataFrame(imputer.transform(test_data[['Current']]), columns=['Current'])

# Re-add the 'Time' column to the preprocessed data
training_data_preprocessed['Time'] = training_data['Time'].values
test_data_preprocessed['Time'] = test_data['Time'].values

# Feature Scaling
scaler = StandardScaler()
training_data_preprocessed['Current'] = scaler.fit_transform(training_data_preprocessed[['Current']])
test_data_preprocessed['Current'] = scaler.transform(test_data_preprocessed[['Current']])

# Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
training_poly = poly.fit_transform(training_data_preprocessed[['Current']])
test_poly = poly.transform(test_data_preprocessed[['Current']])

# Convert polynomial features to DataFrame and add to preprocessed data
training_data_preprocessed = pd.concat([training_data_preprocessed, pd.DataFrame(training_poly, columns=poly.get_feature_names_out(['Current']))], axis=1)
test_data_preprocessed = pd.concat([test_data_preprocessed, pd.DataFrame(test_poly, columns=poly.get_feature_names_out(['Current']))], axis=1)

# Visualization
df = training_data
fault_start, fault_end = 5.1, 5.4
fault_data = df[(df['Time'] >= fault_start) & (df['Time'] <= fault_end)]
normal_data = df[(df['Time'] < fault_start) | (df['Time'] > fault_end)]

plt.figure(figsize=(10, 6))
plt.scatter(normal_data['Time'], normal_data['Current'], c='blue', label='Normal Operation', alpha=0.5)
plt.scatter(fault_data['Time'], fault_data['Current'], c='red', label='Fault (Oscillation)', alpha=0.5)
plt.title('Scatter Plot of Current Over Time in Training Data')
plt.xlabel('Time (s)')
plt.ylabel('Current')
plt.legend()
plt.grid(True)
plt.show()


# Define segmenting and labeling function
def segment_labeling(data, window, overlap, time1, time2):
    index = 0
    windolap = math.floor(window * overlap)
    labels_df = pd.DataFrame(columns=['label'])
    time_series = []
    
    while (index + window) < len(data):
        segment = data.iloc[index: (index + window)]
        label = 'oscillation' if any((time1 <= t <= time2) for t in segment['Time']) else 'normal'
        time_series.append(segment.drop(columns='Time').values.flatten())  # Exclude 'Time' in features
        labels_df = pd.concat([labels_df, pd.DataFrame({'label': [label]})], ignore_index=True)
        index += window - windolap
    
    return time_series, labels_df


# Segment and label the preprocessed training and test data
window = 200
overlap = 0.75
train_X, train_y = segment_labeling(training_data_preprocessed, window, overlap, 5.1, 5.4)
test_X, test_y = segment_labeling(test_data_preprocessed, window, overlap, 2.1, 2.4)

# Convert to arrays for model compatibility
X_train = np.array(train_X)
y_train = train_y['label']
X_test = np.array(test_X)
y_test = test_y['label']

# Apply SMOTE to balance the training data (commented out for now)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Check class distribution after segmentation (without SMOTE)
# print("Training set label distribution:\n", pd.Series(y_train).value_counts())

# Model Training
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model.fit(X_train_smote, y_train_smote)
# rf_model.fit(X_train, y_train)

# Hyperparameter Tuning with GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_leaf': [1, 2, 4, 6, 8],         # Minimum number of samples required to be at a leaf node
}

rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)

# Initialize GridSearchCV with return_train_score=True to get 'mean_train_score'
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='recall_macro', n_jobs=-1, return_train_score=True)
# grid_search.fit(X_train, y_train)
grid_search.fit(X_train_smote, y_train_smote)

# Extract best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best hyperparameters:", best_params)
print("Best cross-validated recall:", best_score)

# Plotting Hyperparameter Tuning Results for min_samples_leaf
results = pd.DataFrame(grid_search.cv_results_)
results['param_min_samples_leaf'] = results['param_min_samples_leaf'].astype(str)

# Plotting mean scores by `min_samples_leaf`
mean_of_means_leaf_train = results.groupby('param_min_samples_leaf')['mean_train_score'].mean()
mean_of_means_leaf_val = results.groupby('param_min_samples_leaf')['mean_test_score'].mean()

plt.figure(figsize=(12, 8))
plt.plot(mean_of_means_leaf_train.index, mean_of_means_leaf_train, label='Mean Train Scores', marker='o')
plt.plot(mean_of_means_leaf_val.index, mean_of_means_leaf_val, label='Mean Validation Scores', marker='o')
plt.xlabel('Min Samples Leaf')
plt.ylabel('Mean Score (Mean of Means)')
plt.title('Min Samples Leaf vs Mean Score')
plt.legend()
plt.show()

# Final Evaluation on Test Data using the Best Estimator
best_estimator = grid_search.best_estimator_
y_predicted_test = best_estimator.predict(X_test)

# Calculate and display classification report at the end
accuracy = accuracy_score(y_test, y_predicted_test)
report = classification_report(y_test, y_predicted_test)

print("\nFinal Accuracy on test data:", accuracy)
print("Final Classification Report:\n", report)