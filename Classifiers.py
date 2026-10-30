#############################################################################################
#######################################  Classifiers ########################################
#############################################################################################

# ========================================== MLPNN ================================================

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import f1_score
import itertools
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report

feature_subsets = {
    "Stability_sel": ['LOC_Os10g38000', 'LOC_Os05g11610', 'LOC_Os01g74140', 'LOC_Os03g13030',
                      'LOC_Os04g40874', 'LOC_Os01g62830', 'LOC_Os04g16130', 'LOC_Os10g41270',
                      'LOC_Os11g13840', 'LOC_Os10g36520'],
    "Boruta":        ['LOC_Os01g04330', 'LOC_Os01g04590', 'LOC_Os01g06590', 'LOC_Os01g12000',
                      'LOC_Os01g14100', 'LOC_Os01g15350', 'LOC_Os01g15640', 'LOC_Os01g18220',
                      'LOC_Os01g19170', 'LOC_Os01g22600'],
    "RFE":           ['LOC_Os01g11280', 'LOC_Os01g74140', 'LOC_Os03g33520', 'LOC_Os03g38210',
                      'LOC_Os03g44290', 'LOC_Os04g16130', 'LOC_Os05g11610', 'LOC_Os09g30250',
                      'LOC_Os10g38000', 'LOC_Os10g41270'],
    "MRMR":          ['LOC_Os03g14010', 'LOC_Os07g42280', 'LOC_Os06g14670', 'LOC_Os01g18220',
                      'LOC_Os08g37444', 'LOC_Os03g09170', 'LOC_Os03g26910', 'LOC_Os09g26920',
                      'LOC_Os04g55920', 'LOC_Os02g52780'],
    "Tree_based":    ['LOC_Os03g09170', 'LOC_Os05g35740', 'LOC_Os06g06880', 'LOC_Os09g23780',
                      'LOC_Os10g07229', 'LOC_Os12g31400', 'LOC_Os11g09230', 'LOC_Os01g12180',
                      'LOC_Os07g15880', 'LOC_Os08g04840'],
    "Genetic":       ['LOC_Os01g51920', 'LOC_Os03g12320', 'LOC_Os04g45810', 'LOC_Os05g32210',
                      'LOC_Os06g51220', 'LOC_Os07g35520', 'LOC_Os07g44610', 'LOC_Os08g35760',
                      'LOC_Os10g30820', 'LOC_Os02g51164']
} ## Drought matrix

feature_subsets = {
    "Stability_sel": ['LOC_Os03g33590', 'LOC_Os02g43410', 'LOC_Os09g37910', 'LOC_Os11g13980'],
    "MRMR":          ['LOC_Os03g14180', 'LOC_Os03g64210', 'LOC_Os04g44830', 'LOC_Os01g04370']
} ## Heat matrix


cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_splits = list(
    cv.split(
        X_train,
        y_train,
        groups=groups_train
    )
)


hidden_layer_sizes = []
for n_layers in range(1, 5):  # 1, 2, 3, 4 layers
    for combo in itertools.product([50, 100], repeat=n_layers):
        hidden_layer_sizes.append(combo)

param_grid = {
    'hidden_layer_sizes': hidden_layer_sizes,
    'alpha': [1e-4, 1e-3, 1e-2],
    'learning_rate_init': [1e-3, 1e-2],
    'activation': ['relu', 'tanh']
}

mlp_results = []

for name, genes in feature_subsets.items():
    X_tr_sub = X_train[genes].values
    y_tr = y_train.values

    mlp = MLPClassifier(max_iter=1000, random_state=42,  verbose=True)
    grid = GridSearchCV(
        mlp,
        param_grid,
        cv=cv_splits,
        scoring='f1_macro',
        n_jobs=-1,
        refit=True,
        verbose=2
    )

    grid.fit(X_tr_sub, y_tr)

    best_params = grid.best_params_
    best_cv_score = grid.best_score_

    best_mlp = grid.best_estimator_
    y_tr_pred = best_mlp.predict(X_tr_sub)
    train_score = f1_score(y_tr, y_tr_pred, average='macro')

    mlp_results.append({
        'Method': name,
        'Best Params': best_params,
        'Train F1': train_score,
        'CV F1': best_cv_score
    })

df_mlp = pd.DataFrame(mlp_results)

import ace_tools_open as tools; tools.display_dataframe_to_user(name="MLP Hyperparameter Tuning Results", dataframe=df_mlp)


reports = {}

for _, row in df_mlp.iterrows():
    method = row['Method']
    best_params = row['Best Params']
    genes = feature_subsets[method]

    clf = MLPClassifier(
        **best_params,
        max_iter=1000,
        random_state=42
    )

    X_tr_sub = X_train[genes].values
    clf.fit(X_tr_sub, y_train.values)

    y_tr_pred = clf.predict(X_tr_sub)
    X_te_sub = X_test[genes].values
    y_te_pred = clf.predict(X_te_sub)

    cm_train = confusion_matrix(y_train, y_tr_pred, labels=clf.classes_)
    cm_test  = confusion_matrix(y_test, y_te_pred,   labels=clf.classes_)

    cr_train = classification_report(y_train, y_tr_pred, target_names=clf.classes_, digits=3)
    cr_test  = classification_report(y_test,  y_te_pred,  target_names=clf.classes_, digits=3)

    reports[method] = {
        'confusion_matrix_train': cm_train,
        'classification_report_train': cr_train,
        'confusion_matrix_test': cm_test,
        'classification_report_test': cr_test
    }

for method, rep in reports.items():
    print(f"\n==================================================================")
    print(f"\n============================= {method} ===========================")
    print(f"\n==================================================================")
    print("Train Confusion Matrix:")
    print(rep['confusion_matrix_train'])
    print("\nTrain Classification Report:")
    print(rep['classification_report_train'])
    print("\nTest Confusion Matrix:")
    print(rep['confusion_matrix_test'])
    print("\nTest Classification Report:")
    print(rep['classification_report_test'])


##======================================== SVM ===============================================



import numpy as np
import pandas as pd
import ast

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)


param_grid = [
    {
        'kernel': ['linear'],
        'C': [0.01, 0.1, 1, 10, 100],
        'class_weight': [None, 'balanced']
    },

    {
        'kernel': ['rbf'],
        'C': [0.01, 0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'class_weight': [None, 'balanced']
    },

    {
        'kernel': ['poly'],
        'C': [0.01, 0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'degree': [2, 3],
        'class_weight': [None, 'balanced']
    }
]


svm_results = []

for name, genes in feature_subsets.items():

    print("\n" + "=" * 80)
    print(f"Running SVM for: {name}")
    print("=" * 80)

    X_tr_sub = X_train[genes].values
    y_tr = y_train.values

    svm = SVC(
        random_state=42
    )

    grid = GridSearchCV(
        estimator=svm,
        param_grid=param_grid,
        cv=cv_splits,
        scoring='f1_macro',
        n_jobs=-1,
        refit=True,
        verbose=2
    )

    grid.fit(X_tr_sub, y_tr)

    best_params = grid.best_params_

    best_cv_score = grid.best_score_

    
    best_svm = grid.best_estimator_

    y_tr_pred = best_svm.predict(X_tr_sub)

    train_accuracy = accuracy_score(y_tr, y_tr_pred)
    train_precision = precision_score(
        y_tr, y_tr_pred, average='macro', zero_division=0
    )
    train_recall = recall_score(
        y_tr, y_tr_pred, average='macro', zero_division=0
    )
    train_f1 = f1_score(
        y_tr, y_tr_pred, average='macro'
    )

    svm_results.append({
        'Method': name,
        'Best Params': best_params,
        'Train Accuracy': train_accuracy,
        'Train Precision': train_precision,
        'Train Recall': train_recall,
        'Train F1': train_f1,
        'CV F1': best_cv_score
    })




df_svm = pd.DataFrame(svm_results)

print("\n\nSVM HYPERPARAMETER TUNING RESULTS")
print("=" * 100)
print(df_svm.to_string(index=False))


svm_reports = {}

for _, row in df_svm.iterrows():

    method = row['Method']
    best_params = row['Best Params']

    genes = feature_subsets[method]

    print("\n" + "=" * 80)
    print(f"FINAL SVM EVALUATION: {method}")
    print("=" * 80)

  

    clf = SVC(
        **best_params,
        random_state=42
    )

    # ------------------------------------------------------------
    # Training data
    # ------------------------------------------------------------

    X_tr_sub = X_train[genes].values
    y_tr = y_train.values

    # ------------------------------------------------------------
    # Test data
    # ------------------------------------------------------------

    X_te_sub = X_test[genes].values
    y_te = y_test.values

    # ------------------------------------------------------------
    # Fit ONLY on training data
    # ------------------------------------------------------------

    clf.fit(X_tr_sub, y_tr)

    # ------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------

    y_tr_pred = clf.predict(X_tr_sub)
    y_te_pred = clf.predict(X_te_sub)

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------

    train_accuracy = accuracy_score(y_tr, y_tr_pred)
    train_precision = precision_score(
        y_tr, y_tr_pred, average='macro', zero_division=0
    )
    train_recall = recall_score(
        y_tr, y_tr_pred, average='macro', zero_division=0
    )
    train_f1 = f1_score(
        y_tr, y_tr_pred, average='macro'
    )

    test_accuracy = accuracy_score(y_te, y_te_pred)
    test_precision = precision_score(
        y_te, y_te_pred, average='macro', zero_division=0
    )
    test_recall = recall_score(
        y_te, y_te_pred, average='macro', zero_division=0
    )
    test_f1 = f1_score(
        y_te, y_te_pred, average='macro'
    )

    # ------------------------------------------------------------
    # Confusion matrices
    # ------------------------------------------------------------

    cm_train = confusion_matrix(
        y_tr,
        y_tr_pred,
        labels=clf.classes_
    )

    cm_test = confusion_matrix(
        y_te,
        y_te_pred,
        labels=clf.classes_
    )

    # ------------------------------------------------------------
    # Classification reports
    # ------------------------------------------------------------

    cr_train = classification_report(
        y_tr,
        y_tr_pred,
        target_names=clf.classes_,
        digits=3,
        zero_division=0
    )

    cr_test = classification_report(
        y_te,
        y_te_pred,
        target_names=clf.classes_,
        digits=3,
        zero_division=0
    )

    # ------------------------------------------------------------
    # Store everything
    # ------------------------------------------------------------

    svm_reports[method] = {

        'best_params': best_params,

        'train_accuracy': train_accuracy,
        'train_precision': train_precision,
        'train_recall': train_recall,
        'train_f1': train_f1,

        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,

        'confusion_matrix_train': cm_train,
        'confusion_matrix_test': cm_test,

        'classification_report_train': cr_train,
        'classification_report_test': cr_test
    }


# ================================================================
# Print final results
# ================================================================

for method, rep in svm_reports.items():

    print("\n")
    print("=" * 80)
    print(f"============================= {method} =============================")
    print("=" * 80)

    print("\nBest parameters:")
    print(rep['best_params'])

    print("\nTRAINING PERFORMANCE")
    print("-" * 50)
    print(f"Accuracy : {rep['train_accuracy']:.3f}")
    print(f"Precision: {rep['train_precision']:.3f}")
    print(f"Recall   : {rep['train_recall']:.3f}")
    print(f"F1       : {rep['train_f1']:.3f}")

    print("\nTEST PERFORMANCE")
    print("-" * 50)
    print(f"Accuracy : {rep['test_accuracy']:.3f}")
    print(f"Precision: {rep['test_precision']:.3f}")
    print(f"Recall   : {rep['test_recall']:.3f}")
    print(f"F1       : {rep['test_f1']:.3f}")

    print("\nTrain Confusion Matrix:")
    print(rep['confusion_matrix_train'])

    print("\nTest Confusion Matrix:")
    print(rep['confusion_matrix_test'])

    print("\nTrain Classification Report:")
    print(rep['classification_report_train'])

    print("Test Classification Report:")
    print(rep['classification_report_test'])


# ================================================================
# Create a compact final SVM results table
# ================================================================

svm_test_results = []

for method, rep in svm_reports.items():

    svm_test_results.append({
        'Method': method,
        'Accuracy': rep['test_accuracy'],
        'Precision': rep['test_precision'],
        'Recall': rep['test_recall'],
        'F1': rep['test_f1']
    })

df_svm_test = pd.DataFrame(svm_test_results)

print("\n\nFINAL SVM TEST RESULTS")
print("=" * 80)
print(df_svm_test.to_string(index=False))


