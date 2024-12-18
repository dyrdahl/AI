# CS 471 AI - Assignment 4
# Implement the Decision tree classification method using Scikit-learn
# By: Shane Dyrdahl
# 10/24/2024

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree

# 2) Split the data into training, validation, and testing sets.
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# 3) Fit a decision tree on the training dataset.
clf = DecisionTreeClassifier(random_state=3)
clf.fit(X_train, y_train)

# 4) Tune at least 2 hyperparameters in the decision tree model
# Hyperparameter to fine tune
param_grid = {
    'max_depth': range(2, 20, 1),
    'min_samples_split': range(2, 20, 1)
}

grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best hyperparameters
best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_

print(f"Best hyperparameters: {best_params}")
print(f"Best estimator: {best_estimator}")

# Get the results of the grid search
results = grid_search.cv_results_

# Aggregate the mean cross-validation accuracy for each max_depth
max_depths = np.array(results['param_max_depth'].data, dtype=int)
mean_scores = np.array(results['mean_test_score'])

# Get unique max_depth values
unique_depths = np.unique(max_depths)

# Calculate the mean accuracy for each unique max_depth
mean_accuracy_per_depth = [
    np.mean(mean_scores[max_depths == depth]) for depth in unique_depths
]

# Aggregate the mean cross-validation accuracy for each min_samples_split
min_samples_splits = np.array(results['param_min_samples_split'].data, dtype=int)

# Get unique min_samples_split values
unique_splits = np.unique(min_samples_splits)

# Calculate the mean accuracy for each unique min_samples_split
mean_accuracy_per_split = [
    np.mean(mean_scores[min_samples_splits == split]) for split in unique_splits
]

# Plot max_depth & min_samples_split vs mean cross-validation accuracy
plt.plot(unique_depths, mean_accuracy_per_depth, marker='o', label='Max Depth')
plt.plot(unique_splits, mean_accuracy_per_split, marker='o', label='Min Samples Split', linestyle='--')
plt.xlabel('Parameter Value')
plt.ylabel('Mean Cross-Validation Accuracy')
plt.title('Performance vs Max Depth and Min Samples Split')
plt.legend(['Max Depth', 'Min Samples Split'])
plt.grid(True)
plt.show()

# 5) Train the model using optimal hyperparameters (found in step 4) on the train + validation data.
y_train_pred = best_estimator.predict(X_train)
final_train_accuracy = accuracy_score(y_train, y_train_pred)
print('\nTraining accuracy: {}'.format(final_train_accuracy))

y_test_pred = best_estimator.predict(X_test)
final_test_accuracy = accuracy_score(y_test, y_test_pred)
print('Testing accuracy: {}'.format(final_test_accuracy))

# Generate a classification report
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))

# 6) Inspect the model by visualizing and interpreting the results
plt.figure(figsize=(20, 10))
plot_tree(best_estimator, filled=True, feature_names=iris.feature_names, class_names=iris.target_names, rounded=True)
plt.title("Decision Tree Visualization")
plt.show()
