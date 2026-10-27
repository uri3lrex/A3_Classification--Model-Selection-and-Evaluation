# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import io
from sklearn.model_selection import train_test_split

# look for final_dataset.csv next to this script; fall back to the current folder
try:
    data_dir = Path(__file__).resolve().parent   # works when run as a .py file
except NameError:
    data_dir = Path.cwd()                        # works in a notebook / Colab

# Read lines and strip any leading Byte Order Mark (BOM) or whitespace because it kept ignoring the id at the start
raw_lines = open(data_dir / "final_dataset.csv", encoding="utf-8-sig").read().splitlines()
lines = [line.strip() for line in raw_lines]

# Identify all start markers starting with '#'
starts = []
for index, line in enumerate(lines):
    if line.startswith('#'):
        starts.append(index)
starts.append(len(lines))

print("Dataset markers found at indexes:", starts[:-1])
for i, idx in enumerate(starts[:-1]):
    print(f"  Dataset {i+1} header: {lines[idx]}")

# Extract and clean lines for Dataset 1 and Dataset 2
dataset1_lines = lines[starts[0]+1 : starts[1]]
cleaned_lines1 = [line.strip('"') for line in dataset1_lines]
dataset1_content = "\n".join(cleaned_lines1)

dataset2_lines = lines[starts[1]+1 : starts[2]]
cleaned_lines2 = [line.strip('"') for line in dataset2_lines]
dataset2_content = "\n".join(cleaned_lines2)

# -----------CHANGE THIS TO SELECT DATASET 1 OR 2----------------
dataset = 1  # <---- CHANGE THIS TO 1 OR 2 TO SELECT DATASET

# Load the selected dataset into a DataFrame
if dataset == 1:
    df = pd.read_csv(io.StringIO(dataset1_content), header=None, comment='#', sep=',')
    print("\n--- Successfully Loaded Dataset 1! ---")
elif dataset == 2:
    df = pd.read_csv(io.StringIO(dataset2_content), header=None, comment='#', sep=',')
    print("\n--- Successfully Loaded Dataset 2! ---")
else:
    print("\nInvalid input. Defaulting to Dataset 1.")
    df = pd.read_csv(io.StringIO(dataset1_content), header=None, comment='#', sep=',')

# Extract features and target columns smoothly and convert them directly to float
X = df.iloc[:, [0, 1]].to_numpy().astype(float)
y = df.iloc[:, 2].to_numpy().astype(float)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(df.head())

"""Using sklearn augment the two features in the dataset with polynomial features
and train a Logistic Regression classifier with L2 penalty added to the cost
function. Use cross-validation to select (i) the maximum order of polynomial to
use and (ii) the weight C given to the penalty in the cost function. Remember
you’ll also need to select the range of C values to consider and the range of
maximum polynomial orders.
Important: It is essential that you present data, including cross-validation plots
and plots of model predictions and the training data, and give clear explanations/analysis to justify your choices. Use the ideas/tools you’ve learned in the
module to help you in this. Be sure to include error bars in cross-validation
plots. Rememberr to use a performance measure appropriate to classification
(so probably not mean square error).
"""

# plotting the data
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
print(f"Train split shape: {X_train.shape}, Test split shape: {X_test.shape}")

# Plot raw data distribution
plt.figure(figsize=(10, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Training Data Distribution')
plt.colorbar(ticks=[-1, 1], label='Class')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
# augment the X1,X2 with polynomial features
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

#
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('polynomial', PolynomialFeatures(include_bias=False)),
    ('logistic', LogisticRegression(penalty='l2', solver='lbfgs', max_iter=2000))
])

parameter_grid = {
    'polynomial__degree': [1,2,3,4,5],
    'logistic__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
}

grid_search = GridSearchCV(pipeline, parameter_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

#display the range of c and polynomials
print("Hyperparameters:")
print(f"Range of Polynomial Degrees considered: {parameter_grid['polynomial__degree']}")
print(f"Range of Regularization Strengths (C) considered: {parameter_grid['logistic__C']}\n")

# getting the best model
best_model = grid_search.best_estimator_

# extract and plot cross-validation results with error Bars
results = grid_search.cv_results_
means = results['mean_test_score']
stds = results['std_test_score']
params = results['params']

# group results by degree to plot separate curves for each polynomial order
c_values = parameter_grid['logistic__C']
degrees = parameter_grid['polynomial__degree']

plt.figure(figsize=(12, 8))
for deg in degrees:
    # find indices where degree equals deg
    indices = [i for i, p in enumerate(params) if p['polynomial__degree'] == deg]
    # sort indices
    indices = sorted(indices, key=lambda i: params[i]['logistic__C'])
    deg_means = [means[i] for i in indices]
    deg_stds = [stds[i] for i in indices]
    plt.errorbar(c_values, deg_means, yerr=deg_stds, fmt='-o', capsize=5, label=f'Degree {deg}')

plt.xscale('log')
plt.xlabel('Regularization Parameter C (Log Scale)')
plt.ylabel('Mean CV Accuracy')
plt.title('Cross-Validation Performance with Error Bars')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()

# Plot Model Predictions and Decision Boundary

#dynamically retrieve the chosen hyperparameters from GridSearchCV
best_i= means.argmax()                              # index of the best mean CV accuracy
threshold = means[best_i] - stds[best_i]             # one standard deviation below the best
candidates = [i for i in range(len(params)) if means[i] >= threshold]  # all configs within one std
simplest = min(candidates, key=lambda i: (params[i]['polynomial__degree'], params[i]['logistic__C']))  # lowest degree, then strongest regularisation
chosen_degree = params[simplest]['polynomial__degree']
chosen_C = params[simplest]['logistic__C']

print(f"Chosen: degree={chosen_degree}, C={chosen_C}, CV acc = {means[simplest]:.4f} ± {stds[simplest]:.4f}")

# Model A: Under-fit (Strictly Linear Baseline)
model_underfit = Pipeline([
    ('scaler', StandardScaler()),
    ('polynomial', PolynomialFeatures(degree=1, include_bias=False)),
    ('logistic', LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=2000))
]).fit(X_train, y_train)

# Model B: The Mathematically Chosen Parity Model
model_chosen = Pipeline([
    ('scaler', StandardScaler()),
    ('polynomial', PolynomialFeatures(degree=chosen_degree, include_bias=False)),
    ('logistic', LogisticRegression(penalty='l2', C=chosen_C, solver='lbfgs', max_iter=2000))
]).fit(X_train, y_train)

best_model_lr = model_chosen

# Model C: Over-fit (Highest Degree, Weak Regularization)
model_overfit = Pipeline([
    ('scaler', StandardScaler()),
    ('polynomial', PolynomialFeatures(degree=5, include_bias=False)),
    ('logistic', LogisticRegression(penalty='l2', C=1000.0, solver='lbfgs', max_iter=2000))
]).fit(X_train, y_train)


# create a dense grid to visualize the decision boundary
x1_min, x1_max = X_train[:, 0].min() - 0.1, X_train[:, 0].max() + 0.1
x2_min, x2_max = X_train[:, 1].min() - 0.1, X_train[:, 1].max() + 0.1
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 500), np.linspace(x2_min, x2_max, 500))
grid_points = np.c_[xx1.ravel(), xx2.ravel()]

# plot other models
fig, axes = plt.subplots(1, 3, figsize=(22, 7), sharex=True, sharey=True)

models = [model_underfit, model_chosen, model_overfit]
titles = [
    "Under-fit Baseline\n(Degree = 1, C = 1.0)",
    f"Chosen Best Model\n(Degree = {chosen_degree}, C = {chosen_C})",
    "Over-fit Complexity\n(Degree = 5, C = 1000.0)"
]


for ax, model, title in zip(axes, models, titles):
    Z = model.predict(grid_points).reshape(xx1.shape)
    ax.contourf(xx1, xx2, Z, levels=[-1.5, 0, 1.5], alpha=0.25, cmap='coolwarm')
    scatter = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', edgecolor='k', alpha=0.7, s=35)
    ax.set_xlabel('Input Feature X1', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(*scatter.legend_elements(), title="Classes", loc="upper left")

axes[0].set_ylabel('Input Feature X2', fontsize=12)
plt.suptitle("Visual Complexity Spectrum: Logistic Regression Boundaries", fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

"""b) Now train a kNN classifier on the data. Use cross-validation to select k, again
presenting data and explanations/analysis to justify your choice. There is no
need to augment the features with polynomial features for a kNN classifier, kNN
can already capture nonlinear decision boundaries. Again, it is essential that
you present data and give clear explanations/analysis to justify your choices.
"""

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# PRODUCING THE GAUSSIAN WEIGHT FUNCTION (From Lecture Page 14)
# since scikit-learn doesn't accept the string 'gaussian', we define it manually:
def gaussian_kernel_gamma1(distances):
    # w = exp(-gamma * d^2), assuming gamma = 1 here
    weights = np.exp(-1.0 * (distances ** 2))
    # Normalize weights so they sum to 1 over the neighborhood
    return weights / np.sum(weights)

def gaussian_kernel_gamma100(distances):
    # w = exp(-gamma * d^2), assuming gamma = 100 here
    weights = np.exp(-100.0 * (distances ** 2))
    # Normalize weights so they sum to 1 over the neighborhood
    return weights / np.sum(weights)

def gaussian_kernel_gamma1000(distances):
    # w = exp(-gamma * d^2), assuming gamma = 1000 here
    weights = np.exp(-1000.0 * (distances ** 2))
    # Normalize weights so they sum to 1 over the neighborhood
    return weights / np.sum(weights)

knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

# grid search
param_grid_knn_1 = {
    'knn__n_neighbors': list(range(1, 50, 2)),
    'knn__weights': ['uniform',  gaussian_kernel_gamma1],
    'knn__metric': ['euclidean', 'manhattan']
}

param_grid_knn_100 = {
    'knn__n_neighbors': list(range(1, 50, 2)),
    'knn__weights': ['uniform', gaussian_kernel_gamma100],
    'knn__metric': ['euclidean', 'manhattan']
}

param_grid_knn_1000 = {
    'knn__n_neighbors': list(range(1, 50, 2)),
    'knn__weights': ['uniform', gaussian_kernel_gamma1000],
    'knn__metric': ['euclidean', 'manhattan']
}

grid_search_knn = GridSearchCV(
    estimator=knn_pipeline,
    param_grid=[param_grid_knn_1, param_grid_knn_100, param_grid_knn_1000],
    cv=5,
    scoring='accuracy'
)

grid_search_knn.fit(X_train, y_train)

# This returns the best model configurations (corrected to use pipeline parameter keys):
print(f"Optimal Number of Neighbors (k): {grid_search_knn.best_params_['knn__n_neighbors']}")
print(f"Optimal Weighting Function: {grid_search_knn.best_params_['knn__weights']}")
print(f"Best CV Accuracy Score: {grid_search_knn.best_score_:.4f}")
print(f"Best metric: {grid_search_knn.best_params_['knn__metric']}")
print(f"Test set score: {grid_search_knn.score(X_test, y_test):.4f}")

# getting the best model
best_model_knn = grid_search_knn.best_estimator_

# extract and plot cross-validation results with error bars
knn_results = grid_search_knn.cv_results_
knn_means = knn_results['mean_test_score']
knn_stds = knn_results['std_test_score']
knn_params = knn_results['params']

# helper function to get clean string representations of weight parameters
def get_weight_name(w):
    return w.__name__ if callable(w) else str(w)

# extract unique values used in the grid search (sorting weights safely by their string names)
k_values = sorted(list(set([p['knn__n_neighbors'] for p in knn_params])))
weights_options = sorted(list(set([p['knn__weights'] for p in knn_params])), key=get_weight_name)
metric_options = sorted(list(set([p['knn__metric'] for p in knn_params])))

plt.figure(figsize=(14, 8))

# plot curves for combinations of weights and metrics
for weight in weights_options:
    weight_label = get_weight_name(weight)
    for metric_val in metric_options:
        label = f"Weights: {weight_label}, Metric: {metric_val.capitalize()}"

        # Find indices matching this configuration, sorted by k
        indices = [i for i, dct in enumerate(knn_params) if dct['knn__weights'] == weight and dct['knn__metric'] == metric_val]
        indices = sorted(indices, key=lambda i: knn_params[i]['knn__n_neighbors'])

        if not indices:
            continue

        # Extract values
        current_k = [knn_params[i]['knn__n_neighbors'] for i in indices]
        current_means = [knn_means[i] for i in indices]
        current_stds = [knn_stds[i] for i in indices]

        plt.errorbar(current_k, current_means, yerr=current_stds, fmt='-o', capsize=4, label=label, alpha=0.8)

plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Mean CV Accuracy')
plt.title('kNN Cross-Validation Performance with Error Bars')
plt.xticks(k_values)
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()

# plot Model Predictions and Decision Boundary for kNN

chosen_k = grid_search_knn.best_params_['knn__n_neighbors']
chosen_weight = grid_search_knn.best_params_['knn__weights']
chosen_metric = grid_search_knn.best_params_['knn__metric']

# model 1: Over-fit (k=1)
model_knn_overfit = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=1, weights='uniform', metric=chosen_metric))
]).fit(X_train, y_train)

# model 2: Mathematically Chosen Model (k=45)
model_knn_chosen = best_model_knn

# model 3: Pseudo-Overfit due to hyperparameter flaw (Large k but hyper-sensitive gamma weight)
model_knn_flawed = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=chosen_k, weights=gaussian_kernel_gamma1000, metric=chosen_metric))
]).fit(X_train, y_train)

# create a dense grid to visualize the decision boundary
x1_min, x1_max = X_train[:, 0].min() - 0.1, X_train[:, 0].max() + 0.1
x2_min, x2_max = X_train[:, 1].min() - 0.1, X_train[:, 1].max() + 0.1
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 500), np.linspace(x2_min, x2_max, 500))
grid_points = np.c_[xx1.ravel(), xx2.ravel()]

fig, axes = plt.subplots(1, 3, figsize=(22, 7), sharex=True, sharey=True)

knn_models = [model_knn_overfit, model_knn_chosen, model_knn_flawed]
knn_titles = [
    "Severe Over-fit\n(k = 1, Weights = Uniform)",
    f"Best CV Model\n(k = {chosen_k}, Weights = {get_weight_name(chosen_weight)})",
    f"Distance-Crushed Neighborhood\n(k = {chosen_k}, Weights = Gaussian Gamma=1000)"
]

# predict over the grid using the best kNN model
for ax, model, title in zip(axes, knn_models, knn_titles):
    Z = model.predict(grid_points).reshape(xx1.shape)
    ax.contourf(xx1, xx2, Z, levels=[-1.5, 0, 1.5], alpha=0.25, cmap='coolwarm')
    scatter = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', edgecolor='k', alpha=0.7, s=35)
    ax.set_xlabel('Input Feature X1', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(*scatter.legend_elements(), title="Classes", loc="upper left")

axes[0].set_ylabel('Input Feature X2', fontsize=12)
plt.suptitle("Visual Complexity Spectrum: kNN Decision Boundaries", fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# performance measure classifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

for name, model in [("Logistic Regression", best_model_lr), ("kNN Classifier", best_model_knn)]:
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print(f"\n--- {name} Metrics ---")
    print(f"Accuracy:  {accuracy_score(y_test, preds):.4f}")
    print(f"Precision: {precision_score(y_test, preds, pos_label=1):.4f}")
    print(f"Recall:    {recall_score(y_test, preds, pos_label=1):.4f}")
    print(f"F1 Score:  {f1_score(y_test, preds, pos_label=1):.4f}")
    print(f"ROC AUC:   {roc_auc_score(y_test, probs):.4f}")

"""(c) Calculate the confusion matrices for your trained Logistic Regression and kNN
classifier. Also calculate the confusion matrix for one or more baseline classifiers
of your choice e.g. one that always predicts the most frequent class in the training
data and/or one that makes random predictions.

(d) Plot the ROC curves for your trained Logistic Regression and kNN classifiers.
Also plot the point(s) oin the ROC plot corresponding to the baseline classifiers.
Be sure to include enough points in the ROC curves to allow the detailed shape
to be seen.
"""

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt

# define helper function
def fpr_tpr(cm):
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    return fpr, tpr

# train baseline classifiers on the training set
majority = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
rand = DummyClassifier(strategy="uniform", random_state=42).fit(X_train, y_train)

# collect the models to evaluate
models = {
    "Logistic regression": best_model_lr,
    "kNN": best_model_knn,
    "Baseline: most frequent": majority,
    "Baseline: random": rand
}

# confusion matrices on the held-out test set
cms = {m: confusion_matrix(y_test, est.predict(X_test), labels=[-1, 1]) for m, est in models.items()}
for m, cm in cms.items():
    print(f"\n{m} | test accuracy {np.trace(cm) / cm.sum():.3f}")
    print(pd.DataFrame(cm, index=["Actual -1", "Actual +1"], columns=["Pred -1", "Pred +1"]))

# ROC curves on the held-out test set
plt.figure(figsize=(8, 7))
for m in ["Logistic regression", "kNN"]:
    est = models[m]
    p = est.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, p, pos_label=1)
    plt.plot(fpr, tpr, lw=2, label=f"{m} (AUC = {roc_auc_score(y_test, p):.3f})")

for m, marker in [("Baseline: most frequent", "s"), ("Baseline: random", "X")]:
    plt.scatter(*fpr_tpr(cms[m]), marker=marker, s=120, zorder=5, label=m)

plt.plot([0, 1], [0, 1], "k--", label="Random classifier")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title(f"Dataset {dataset}: ROC on held-out test set")
plt.legend(loc="lower right")
plt.grid(True, ls="--")
plt.show()

print("-----END OF ASSIGNMENT-----")
print("Please check line 40 for the dataset selection. Change the value of 'dataset' to 1 or 2 to switch between datasets.")
print("-----END OF COMPARISON-----")

"""(e) Using the data from (c) and (d) evaluate and compare the performance of the
Logistic Regression, kNN and baseline classifiers. Is one classifier significantly
better or worse than the other? How do their ROC curves compare with that
of a random classifier. Which classifier, if any, would you recommend be used?
Explain the reasoning behind you recommendation.
"""