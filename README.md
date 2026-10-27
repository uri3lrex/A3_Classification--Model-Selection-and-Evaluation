Data: two datasets (1452 and 552 points), 2 features, ±1 target. 75/25 stratified train/test split; the test set is never used for tuning.

What I did

Logistic regression with polynomial features and L2 penalty: grid search over degree (1 to 5) and C (10⁻³ to 10³), scored by 5-fold cross-validated accuracy.
kNN: grid search over k (1 to 49, odd), weighting (uniform and Gaussian-kernel) and distance metric (Euclidean, Manhattan).
Evaluated on the held-out test set with confusion matrices and ROC/AUC, compared with two baselines (most frequent class, random).
Judged whether a model difference was meaningful by comparing it with the cross-validation standard deviation and the test-set size.

Results

Dataset 1: no usable signal. Every configuration scores about 0.674 in CV, equal to the majority baseline. Both models collapse to predicting +1 everywhere (test accuracy 0.675, AUC 0.52 and 0.51). Conclusion: no classifier is worth using.
Dataset 2: the classifiers work. Logistic regression (degree 4, C = 10): CV 0.978 ± 0.016, test accuracy 0.964, AUC 0.998. kNN (k = 9): CV 0.983 ± 0.012, test accuracy 0.949, AUC 0.989. Baseline: 0.783. The gap between the two models is 2 mistakes out of 138, which is within noise.