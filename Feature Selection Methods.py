#########################################################################################################
####################################### Feature Selection Methods #######################################
#########################################################################################################

#=========================== Stability Selection ==========================
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier

n_bootstraps = 1000
sample_frac = 0.75
n_samples, n_features = X_train.shape

selection_counts = np.zeros(n_features, dtype=int)

for _ in range(n_bootstraps):
    subsample_idx = np.random.choice(
        np.arange(n_samples),
        size=int(sample_frac * n_samples),
        replace=False
    )
    X_sub = X_train.values[subsample_idx]
    y_sub = y_train.values[subsample_idx]
    
    l1_clf = LogisticRegression(
        penalty='l1', solver='liblinear',
        C=0.5, class_weight='balanced',
        random_state=None, max_iter=1000
    )
    l1_clf.fit(X_sub, y_sub)
    
    selection_counts += (l1_clf.coef_[0] != 0).astype(int)

selection_freq = selection_counts / n_bootstraps

top10_idx = np.argsort(selection_freq)[::-1][:10]  #4 features (genes) for heat stress
selected_genes = [feature_names[i] for i in top10_idx]
print("Top 10 genes by stability frequency:\n", selected_genes)

X_train_sel = X_train[selected_genes].values
X_test_sel  = X_test[selected_genes].values

models = {
    'Logistic (L2)': LogisticRegression(
        penalty='l2', solver='liblinear', C=0.5,
        class_weight='balanced', random_state=42, max_iter=1000
    ),
    'RidgeClassifier': RidgeClassifier(
        class_weight='balanced', random_state=42
    )
}

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
results = []
for name, clf in models.items():
    # 4a) Train‐F1
    clf.fit(X_train_sel, y_train)
    y_tr_pred = clf.predict(X_train_sel)
    train_f1 = f1_score(y_train, y_tr_pred, average='macro')
    
    cv_scores = cross_val_score(
        clf, X_train_sel, y_train,
        cv=cv_splits,
        scoring='f1_macro',
        n_jobs=-1
    )
    cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
    
    y_te_pred = clf.predict(X_test_sel)
    test_f1 = f1_score(y_test, y_te_pred, average='macro')
    
    results.append({
        'Model': name,
        'Train F1': train_f1,
        'CV F1 (mean±std)': f"{cv_mean:.3f} ± {cv_std:.3f}",
        'Test F1': test_f1
    })

df_results = pd.DataFrame(results)
df_results

#========================================= Boruta =========================================

import numpy as np
import pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score

feature_names = X_train.columns.tolist()

rf = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

boruta = BorutaPy(
    estimator=rf,
    n_estimators='auto',
    perc=100,        # threshold percent
    alpha=0.05,      # significance level
    two_step=True,   # stricter selection
    random_state=42,
    verbose=0
)
boruta.fit(X_train.values, y_train.values)

ranks = boruta.ranking_          # 1 = confirmed, 2+ = tentative or rejected
feat_rank = sorted(zip(feature_names, ranks), key=lambda x: x[1])
top10 = [f for f, r in feat_rank[:10]]
print("Boruta top-10 features:\n", top10)

X_tr_sel  = X_train[top10].values
X_te_sel  = X_test[top10].values

models = {
    'Logistic (L2)': LogisticRegression(
        penalty='l2', solver='liblinear',
        class_weight='balanced', random_state=42, max_iter=1000
    ),
    'RidgeClassifier': RidgeClassifier(
        class_weight='balanced', random_state=42
    )
}

results = []
for name, clf in models.items():
    clf.fit(X_tr_sel, y_train)
    y_tr_pred = clf.predict(X_tr_sel)
    train_f1 = f1_score(y_train, y_tr_pred, average='macro')
    
    cv_scores = cross_val_score(
        clf, X_tr_sel, y_train, cv=cv_splits,
        scoring='f1_macro', n_jobs=-1
    )
    cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
    
    y_te_pred = clf.predict(X_te_sel)
    test_f1 = f1_score(y_test, y_te_pred, average='macro')
    
    results.append({
        'Model': name,
        'Train F1': train_f1,
        'CV F1 (mean±std)': f"{cv_mean:.3f} ± {cv_std:.3f}",
        'Test F1': test_f1
    })

df_boruta = pd.DataFrame(results)
print(df_boruta)

#======================================= Recursive Feature Elimination with Cross Validation(RFECV) ==============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score



rfe = RFE(
    estimator=LogisticRegression(
        penalty='l2', solver='liblinear', C=0.5,
        class_weight='balanced', random_state=42, max_iter=1000
    ),
    n_features_to_select=10,
    step=0.1,            # eliminate 10% of features each round
    verbose=0
)
rfe.fit(X_train, y_train)

top10_rfe = [feat for feat, keep in zip(feature_names, rfe.support_) if keep]
print("RFE top-10 features:\n", top10_rfe)

X_tr_rfe = X_train[top10_rfe].values
X_te_rfe = X_test [top10_rfe].values

models = {
    'Logistic (L2)': LogisticRegression(
        penalty='l2', solver='liblinear', C=0.5,
        class_weight='balanced', random_state=42, max_iter=1000
    ),
    'RidgeClassifier': RidgeClassifier(
        class_weight='balanced', random_state=42
    )
}

rfe_results = []
for name, clf in models.items():
    clf.fit(X_tr_rfe, y_train)
    train_f1 = f1_score(y_train, clf.predict(X_tr_rfe), average='macro')
    
    cv_scores = cross_val_score(
        clf, X_tr_rfe, y_train,
        cv=cv_splits, scoring='f1_macro', n_jobs=-1
    )
    cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
    
    test_f1 = f1_score(y_test, clf.predict(X_te_rfe), average='macro')
    
    rfe_results.append({
        'Model': name,
        'Train F1': train_f1,
        'CV F1 (mean±std)': f"{cv_mean:.3f} ± {cv_std:.3f}",
        'Test F1': test_f1
    })

df_rfe = pd.DataFrame(rfe_results)
print("\nRFE results:\n", df_rfe)


rfecv = RFECV(
    estimator=LogisticRegression(
        penalty='l2', solver='liblinear', C=0.5,
        class_weight='balanced', random_state=42, max_iter=1000
    ),
    step=0.1,
    cv=cv_splits,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=0
)
rfecv.fit(X_train, y_train)

print(f"\nRFECV selected {rfecv.n_features_} features (best CV score).")

top_rfecv = [feat for feat, keep in zip(feature_names, rfecv.support_) if keep]
print("RFECV features:\n", top_rfecv)

X_tr_rfecv = X_train[top_rfecv].values
X_te_rfecv = X_test [top_rfecv].values

rfecv_results = []
for name, clf in models.items():
    clf.fit(X_tr_rfecv, y_train)
    train_f1 = f1_score(y_train, clf.predict(X_tr_rfecv), average='macro')
    cv_scores = cross_val_score(clf, X_tr_rfecv, y_train, cv=cv_splits, scoring='f1_macro', n_jobs=-1)
    cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
    test_f1 = f1_score(y_test, clf.predict(X_te_rfecv), average='macro')
    rfecv_results.append({
        'Model': name,
        'Train F1': train_f1,
        'CV F1 (mean±std)': f"{cv_mean:.3f} ± {cv_std:.3f}",
        'Test F1': test_f1
    })

df_rfecv = pd.DataFrame(rfecv_results)
print("\nRFECV results:\n", df_rfecv)


#============================== Mutual Information- Minimum Redundancy Maximum Relevance (MImRMR)================

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score


mi = mutual_info_classif(X_train, y_train, discrete_features=False, random_state=42)
mi_idx = np.argsort(mi)[::-1][:10]           #4 features (genes) for heat stress
mi_genes = [feature_names[i] for i in mi_idx]
print("Top-10 by Mutual Information:\n", mi_genes)

def evaluate_on_features(gene_list, label):
    """
    Subset to gene_list, then train+CV+test both models and return a DataFrame.
    """
    X_tr = X_train[gene_list].values
    X_te = X_test [gene_list].values

    results = []
    for name, clf in models.items():
        clf.fit(X_tr, y_train)
        train_f1 = f1_score(y_train, clf.predict(X_tr), average='macro')
        
        cv_scores = cross_val_score(clf, X_tr, y_train,
                                    cv=cv_splits, scoring='f1_macro', n_jobs=-1)
        cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
        
        test_f1 = f1_score(y_test, clf.predict(X_te), average='macro')
        
        results.append({
            'Method': label,
            'Model': name,
            'Train F1': train_f1,
            'CV F1 (mean±std)': f"{cv_mean:.3f} ± {cv_std:.3f}",
            'Test F1': test_f1
        })
    return pd.DataFrame(results)

df_mi = evaluate_on_features(mi_genes, 'Mutual Information')
print(df_mi)


mi_all = mutual_info_classif(X_train, y_train, discrete_features=False, random_state=42)
pairwise_mi = np.zeros((len(feature_names), len(feature_names)))
for i in range(len(feature_names)):
    pairwise_mi[i, :] = mutual_info_classif(
        X_train, X_train.iloc[:, i], discrete_features=False, random_state=42
    )

selected = []
candidates = set(range(len(feature_names)))
selected.append(mi_all.argmax())
candidates.remove(selected[0])

while len(selected) < 20:
    scores = {}
    for j in candidates:
        relevance = mi_all[j]
        redundancy = np.mean([pairwise_mi[j, s] for s in selected])
        scores[j] = relevance - redundancy
    best = max(scores, key=scores.get)
    selected.append(best)
    candidates.remove(best)

mrmr_genes = [feature_names[i] for i in selected]
print("Top-10 by mRMR:\n", mrmr_genes)

df_mrmr = evaluate_on_features(mrmr_genes, 'mRMR')
print(df_mrmr)


df_all = pd.concat([df_mi, df_mrmr], ignore_index=True)
print("\nSummary of MI vs mRMR selection:")
print(df_all)

#================================ Embedded gradient boosting feature selection (EGBFS) =================================

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score


gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gb.fit(X_train, y_train)

importances = gb.feature_importances_
ranked_idx = np.argsort(importances)[::-1]
top10_idx = ranked_idx[:10]
top10_genes = [feature_names[i] for i in top10_idx]
print("Top 10 genes by GradientBoosting importances:\n", top10_genes)

X_tr_gb = X_train[top10_genes].values
X_te_gb = X_test [top10_genes].values

results = []
for name, clf in models.items():
    clf.fit(X_tr_gb, y_train)
    train_f1 = f1_score(y_train, clf.predict(X_tr_gb), average='macro')
    
    cv_scores = cross_val_score(
        clf, X_tr_gb, y_train,
        cv=cv_splits,
        scoring='f1_macro',
        n_jobs=-1
    )
    cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
    
    test_f1 = f1_score(y_test, clf.predict(X_te_gb), average='macro')
    
    results.append({
        'Model': name,
        'Train F1': train_f1,
        'CV F1 (mean±std)': f"{cv_mean:.3f} ± {cv_std:.3f}",
        'Test F1': test_f1
    })

df_gb = pd.DataFrame(results)
print(df_gb)

#================================================ genetic Algorithm ===================================================

from sklearn_genetic import GAFeatureSelectionCV
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score
import numpy as np, pandas as pd

ga = GAFeatureSelectionCV(
    estimator=LogisticRegression(
        penalty='l2', solver='liblinear', C=0.5,
        class_weight='balanced', random_state=42, max_iter=1000
    ),
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring='f1_macro',
    population_size=80,
    generations=35, #30
    crossover_probability=0.5,
    mutation_probability=0.2,
    keep_top_k=10,             # force the GA to return exactly 10 features
    n_jobs=-1,
    verbose=True,
    max_features=15
)

ga.fit(X_train.values, y_train.values)

mask = ga.get_support()        # boolean mask of length n_features
selected_genes = [feat for feat, keep in zip(feature_names, mask) if keep]
print("GA‐selected features:", selected_genes)

X_tr_ga = X_train[selected_genes].values
X_te_ga = X_test [selected_genes].values

models = {
    'Logistic (L2)': LogisticRegression(
        penalty='l2', solver='liblinear', C=0.5,
        class_weight='balanced', random_state=42, max_iter=1000
    ),
    'RidgeClassifier': RidgeClassifier(
        class_weight='balanced', random_state=42
    )
}

results = []
for name, clf in models.items():
    clf.fit(X_tr_ga, y_train)
    train_f1 = f1_score(y_train, clf.predict(X_tr_ga), average='macro')
    cv = cross_val_score(clf, X_tr_ga, y_train, cv=cv_splits, scoring='f1_macro', n_jobs=-1)
    test_f1 = f1_score(y_test, clf.predict(X_te_ga), average='macro')
    results.append({
        'Model': name,
        'Train F1': train_f1,
        'CV F1 (mean±std)': f"{cv.mean():.3f} ± {cv.std():.3f}",
        'Test F1': test_f1
    })

df_ga = pd.DataFrame(results)
print(df_ga)

