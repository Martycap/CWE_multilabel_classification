# Predicting CWE categories from CVE descriptions

This project addresses the multilabel classification of CVE descriptions into one or more CWE categories.

The final dataset contains approximately **292,000 CVEs** and **150 CWE classes** after hierarchy-aware label cleaning and aggregation.

## Models

Three models are compared using the same TF-IDF representation:

- Logistic Regression One-vs-Rest
- Linear SVM One-vs-Rest
- Multilayer Perceptron (MLP)

Logistic Regression and Linear SVM use balanced class weights.

## Experimental Protocol

The dataset is split using multilabel stratification into:

- **85% Development Set**
- **15% independent Test Set**

A **5-fold multilabel-stratified cross-validation** is performed on the Development Set.

The same folds are used for all models.

For each fold:

1. TF-IDF is fitted only on the training fold.
2. The model is trained on the training fold.
3. Hyperparameters are evaluated on the validation fold.
4. The prediction threshold is selected on the validation fold.

The best configuration is selected mainly according to the mean **Macro-F1**.

After model selection, each model is retrained on the complete Development Set and evaluated once on the independent Test Set.

The Test Set is never used for hyperparameter or threshold selection.

## TF-IDF

Main configuration:

```text
N-grams        (1, 2)
min_df         2
max_df         0.95
Max features   50,000
Sublinear TF   True
Stop words     English
URL normalization enabled
```

The TF-IDF vectorizer is fitted only on training data to prevent data leakage.

## Hyperparameters

### Logistic Regression

```text
C = {0.5, 1.0, 2.0}
Best C = 2.0
Penalty = L2
Class weight = balanced
```

### Linear SVM

```text
C = {0.5, 1.0, 2.0}
Best C = 1.0
Penalty = L2
Loss = squared hinge
Class weight = balanced
```

### MLP

```text
Hidden layer = {128, 256}
Best hidden layer = 256
Alpha = 0.001
Learning rate = 0.001
Batch size = 256
Max iterations = 60
Activation = ReLU
Optimizer = Adam
Early stopping = enabled
```

The MLP directly uses the sparse TF-IDF representation without dimensionality reduction.



## Run

Run all experiments:

```bash
python -m src.models.kfold_tfidf_experiments
```

Run a single model:

```bash
python -m src.models.kfold_tfidf_experiments --model logistic_regression
python -m src.models.kfold_tfidf_experiments --model linear_svm
python -m src.models.kfold_tfidf_experiments --model mlp
```
