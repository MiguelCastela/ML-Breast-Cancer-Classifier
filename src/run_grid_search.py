import numpy as np
import pandas as pd
import sys
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from grid_search import (
    grid_search_knn,
    grid_search_svm_custom,
    grid_search_dt,
    grid_search_adaboost,
    grid_search_adaboost_custom,
    grid_search_random_forest,
    grid_search_svm_rbf,
    grid_search_svm_linear


)
from sklearn.metrics import accuracy_score, f1_score  
from sklearn.model_selection import StratifiedKFold

#KNN (trimmed)
KNN_K_LIST = [1, 3, 5, 7, 9, 11, 13, 15, 17]

# C list for SVM (both Linear and RBF) (trimmed)
SVM_C_LIST_LINEAR = [0.03125, 0.125, 0.5, 1.0, 2.0, 8.0, 32.0]
SVM_C_LIST_RBF = [0.03125, 0.125, 0.5, 1.0, 2.0, 8.0, 32.0]

# Gamma list: RBF SVM (trimmed, centered around reasonable ranges)
SVM_GAMMA_LIST = [0.0625, 0.125, 0.25, 0.5, 1.0]

# Decision Tree (trimmed)
DT_DEPTH_LIST = [1, 2, 3, 4, 5, 7, 10, 15]
DT_CRITERION_LIST = ['gini', 'entropy']

# AdaBoost (trimmed)
AB_N_EST_LIST = [10, 25, 50, 75]
AB_LR_LIST = [0.05, 0.1, 0.2]

# Random Forest (trimmed)
RF_N_EST_LIST = [15, 25, 50, 100]
RF_MAX_DEPTH_LIST = [1, 2, 3, 4, 5, 7, 10, 15]

def get_y_arrays_from_ixs(ixs):
    """Converts (ixHealthy, ixCancer) tuple to a 0/1 label array."""
    ixHealthy, ixCancer = ixs
    y = np.concatenate([np.zeros(len(ixHealthy[0])), np.ones(len(ixCancer[0]))])
    return y

def search_knn_wrapper(X_train, X_val, y_train_ixs, y_val_ixs, k_list):
    """Wrapper for grid_search_knn_validated."""
    y_train = get_y_arrays_from_ixs(y_train_ixs)
    y_val = get_y_arrays_from_ixs(y_val_ixs)
    
    # We must construct a dummy error matrix for the validated function
    errMat_dummy = np.zeros((1, len(k_list))) 
    
    # Assuming grid_search_knn_validated is available and correctly implemented
    # We call it with a dummy run_index=0
    avgError, stdError, optK = grid_search_knn(
        X_train, y_train, X_val, y_val, np.array(k_list)
    )
    print(f"  -> Optimal K: {optK}")
    return optK

def search_svm_linear_wrapper(X_train, X_val, y_train_ixs, y_val_ixs, C_list):
    """Wrapper for Linear SVM C optimization using grid_search_svm_linear."""
    y_train = get_y_arrays_from_ixs(y_train_ixs)
    y_val = get_y_arrays_from_ixs(y_val_ixs)
    CsP2 = np.log2(C_list)
    
    avgError, stdError, optC = grid_search_svm_linear(
        X_train, y_train, X_val, y_val, CsP2
    )
    print(f"  -> Optimal C (Linear SVM): {optC:.4f}")
    return optC

def search_svm_rbf_wrapper(X_train, X_val, y_train_ixs, y_val_ixs, C_list, gamma_list):
    """Use grid_search_svm_rbf for RBF SVM C and Gamma optimization."""
    y_train = get_y_arrays_from_ixs(y_train_ixs)
    y_val = get_y_arrays_from_ixs(y_val_ixs)
    CsP2 = np.log2(C_list)
    GammasP2 = np.log2(gamma_list)

    avgError, stdError, optC, optG = grid_search_svm_rbf(
        X_train, y_train, X_val, y_val, CsP2, GammasP2
    )
    print(f"  -> Optimal C (RBF SVM): {optC:.4f}, Optimal Gamma: {optG:.4f}")
    return optC, optG


def search_svm_custom_wrapper(X_train, X_val, y_train_ixs, y_val_ixs, C_list):
    """Wrapper for Custom SVM C optimization using grid_search_svm_custom.
    Uses the same C list as linear SVM (converted to log2 exponents).
    """
    y_train = get_y_arrays_from_ixs(y_train_ixs)
    y_val = get_y_arrays_from_ixs(y_val_ixs)
    CsP2 = np.log2(C_list)

    avgError, stdError, optC = grid_search_svm_custom(
        X_train, y_train, X_val, y_val, CsP2
    )
    print(f"  -> Optimal C (Custom SVM): {optC:.4f}")
    return optC

def search_optimal_params_kfold(train_data, all_features, top5_roc, top5_kruskall,
                                k=5,
                                knn_k_list=KNN_K_LIST,
                                svm_C_list_linear=SVM_C_LIST_LINEAR,
                                svm_C_list_rbf=SVM_C_LIST_RBF,
                                svm_gamma_list=SVM_GAMMA_LIST,
                                dt_depth_list=DT_DEPTH_LIST,
                                dt_criterion_list=DT_CRITERION_LIST,
                                ab_n_est_list=AB_N_EST_LIST,
                                ab_lr_list=AB_LR_LIST,
                                rf_n_est_list=RF_N_EST_LIST,
                                rf_max_depth_list=RF_MAX_DEPTH_LIST,
                                progress_every=50,
                                enable_custom_svm=False,
                                X_pca_all=None,
                                LD1_all=None,
                                val_data=None,
                                X_pca_val=None,
                                LD1_val=None):
    """
    K-fold hyperparameter search using training (+ validation if provided).
    If `val_data` is passed, it is concatenated with `train_data` before CV.
    For PCA/LDA arrays, pass corresponding validation arrays (`X_pca_val`, `LD1_val`)
    so they can be concatenated to match the combined labels length.
    Returns best params per feature set by lowest average validation error.
    Note: PCA/LDA not included here unless fold-specific arrays are provided.
    """
    # Combine train + validation data if available
    if val_data is not None and isinstance(val_data, pd.DataFrame):
        try:
            train_data = pd.concat([train_data, val_data], ignore_index=True)
        except Exception:
            # Fallback: if columns differ, align common columns
            common_cols = [c for c in train_data.columns if c in val_data.columns]
            train_data = pd.concat([
                train_data[common_cols], val_data[common_cols]
            ], ignore_index=True)
    feature_sets_to_test = {
        "All Features": all_features,
        "ROC-AUC Top 5": top5_roc,
        "Kruskal-Wallis Top 5": top5_kruskall
    }
    # Add PCA/LDA analogous to run_all_classifiers, using provided arrays
    if X_pca_all is not None and isinstance(X_pca_all, np.ndarray):
        # If validation PCA is provided and val_data used, concatenate to align sizes
        if val_data is not None and X_pca_val is not None and isinstance(X_pca_val, np.ndarray):
            try:
                X_pca_all = np.vstack([X_pca_all, X_pca_val])
            except Exception:
                pass
        feature_sets_to_test["PCA Features"] = [f"PC{i+1}" for i in range(X_pca_all.shape[1])]
    if LD1_all is not None and isinstance(LD1_all, np.ndarray):
        if val_data is not None and LD1_val is not None and isinstance(LD1_val, np.ndarray):
            try:
                LD1_all = np.concatenate([LD1_all, LD1_val], axis=0)
            except Exception:
                pass
        feature_sets_to_test["LDA Features"] = ["LD1"]

    y_all = (train_data["Classification"].values == "Cancer").astype(int)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    optimal_params = {}
    metrics_summary = {}


    for name, features in feature_sets_to_test.items():
        if features is None or len(features) == 0:
            print(f"\nSkipping k-fold search for {name}: Features not available.")
            continue

        print(f"\n========================================================")
        print(f"K-FOLD HYPERPARAMETER SEARCH FOR: {name} (Features: {features})")
        print(f"========================================================")
        metrics_summary[name] = {}

        # Prepare full X matrix once
        if name == "PCA Features" and X_pca_all is not None:
            X_all = X_pca_all
        elif name == "LDA Features" and LD1_all is not None:
            X_all = LD1_all
        else:
            X_all = train_data[features].values
        # Ensure 2D shape for single-feature arrays (e.g., LDA LD1)
        if isinstance(X_all, np.ndarray) and X_all.ndim == 1:
            X_all = X_all.reshape(-1, 1)

        # Helper to average F1-based errors across folds for a given evaluator
        def avg_error_over_folds(eval_fn):
            errs = []
            for train_idx, val_idx in skf.split(X_all, y_all):
                X_tr, X_va = X_all[train_idx], X_all[val_idx]
                y_tr, y_va = y_all[train_idx], y_all[val_idx]
                err = eval_fn(X_tr, y_tr, X_va, y_va)
                errs.append(err)
            return float(np.mean(errs)), float(np.std(errs))

        # Helper to compute train/val avg and std for F1 and Acc across folds
        def avg_train_val_metrics_over_folds(build_fn):
            f1s_tr, accs_tr = [], []
            f1s_va, accs_va = [], []
            for train_idx, val_idx in skf.split(X_all, y_all):
                X_tr, X_va = X_all[train_idx], X_all[val_idx]
                y_tr, y_va = y_all[train_idx], y_all[val_idx]
                clf = build_fn(X_tr, y_tr)
                # Train predictions (on training fold)
                y_pred_tr = clf.predict(X_tr)
                f1s_tr.append(f1_score(y_tr, y_pred_tr))
                accs_tr.append(accuracy_score(y_tr, y_pred_tr))
                # Validation predictions (on held-out fold)
                y_pred_va = clf.predict(X_va)
                f1s_va.append(f1_score(y_va, y_pred_va))
                accs_va.append(accuracy_score(y_va, y_pred_va))
            return {
                'train_avg_f1': float(np.mean(f1s_tr)),
                'train_std_f1': float(np.std(f1s_tr)),
                'train_avg_acc': float(np.mean(accs_tr)),
                'train_std_acc': float(np.std(accs_tr)),
                'val_avg_f1': float(np.mean(f1s_va)),
                'val_std_f1': float(np.std(f1s_va)),
                'val_avg_acc': float(np.mean(accs_va)),
                'val_std_acc': float(np.std(accs_va)),
            }

        # 1. KNN
        best_err = np.inf
        best_k = None
        print(f"  KNN grid: {len(knn_k_list)} k values")
        start_ts = time.monotonic()
        done = 0
        total = len(knn_k_list)
        for k_val in knn_k_list:
            def _eval_knn(X_tr, y_tr, X_va, y_va, k=k_val):
                clf = KNeighborsClassifier(n_neighbors=k)
                clf.fit(X_tr, y_tr)
                y_pred = clf.predict(X_va)
                return 100 * (1 - f1_score(y_va, y_pred))
            avg_err, std_err = avg_error_over_folds(_eval_knn)
            done += 1
            elapsed = time.monotonic() - start_ts
            eta = (elapsed / max(done, 1)) * (total - done)
            print(f"    k={k_val} avg_err={avg_err:.2f}% (progress {done}/{total}, elapsed {elapsed:.1f}s, ETA {eta:.1f}s)")
            sys.stdout.flush()
            if avg_err < best_err:
                best_err, best_k = avg_err, k_val
        optimal_params[f'KNN_{name}'] = best_k
        # Print avg F1/Acc for best K
        def _build_knn(Xtr, ytr, k=best_k):
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(Xtr, ytr)
            return knn
        knn_mets = avg_train_val_metrics_over_folds(lambda Xtr, ytr: _build_knn(Xtr, ytr, best_k))
        print(
            f"  -> Optimal K: {best_k} | "
            f"train F1={knn_mets['train_avg_f1']*100:.2f}%±{knn_mets['train_std_f1']*100:.2f}% "
            f"train Acc={knn_mets['train_avg_acc']*100:.2f}%±{knn_mets['train_std_acc']*100:.2f}% | "
            f"val F1={knn_mets['val_avg_f1']*100:.2f}%±{knn_mets['val_std_f1']*100:.2f}% "
            f"val Acc={knn_mets['val_avg_acc']*100:.2f}%±{knn_mets['val_std_acc']*100:.2f}% "
            f"(avg 1-F1 err {best_err:.2f}%)"
        )
        metrics_summary[name]['KNN'] = knn_mets

        # 2. Linear SVM (C)
        best_err = np.inf
        best_c_lin = None
        print(f"  Linear SVM grid: {len(svm_C_list_linear)} C values")
        c_counter = 0
        start_ts = time.monotonic()
        total = len(svm_C_list_linear)
        for C in svm_C_list_linear:
            def _eval_svm_lin(X_tr, y_tr, X_va, y_va, C_val=C):
                clf = SVC(kernel="linear", C=C_val, random_state=42)
                clf.fit(X_tr, y_tr)
                y_pred = clf.predict(X_va)
                return 100 * (1 - f1_score(y_va, y_pred))
            avg_err, std_err = avg_error_over_folds(_eval_svm_lin)
            c_counter += 1
            if c_counter % max(1, progress_every//5) == 0 or c_counter == 1:
                elapsed = time.monotonic() - start_ts
                eta = (elapsed / max(c_counter, 1)) * (total - c_counter)
                print(f"    C={C:.4f} (progress {c_counter}/{total}) avg_err={avg_err:.2f}% (elapsed {elapsed:.1f}s, ETA {eta:.1f}s)")
                sys.stdout.flush()
            if avg_err < best_err:
                best_err, best_c_lin = avg_err, C
        
        optimal_params[f'SVM_Linear_{name}'] = best_c_lin
        # Print avg F1/Acc for best Linear SVM
        def _build_svm_lin(Xtr, ytr, C_val=best_c_lin):
            clf = SVC(kernel="linear", C=C_val, random_state=42)
            clf.fit(Xtr, ytr)
            return clf
        lin_mets = avg_train_val_metrics_over_folds(lambda Xtr, ytr: _build_svm_lin(Xtr, ytr, best_c_lin))
        print(
            f"  -> Optimal C (Linear SVM): {best_c_lin} | "
            f"train F1={lin_mets['train_avg_f1']*100:.2f}%±{lin_mets['train_std_f1']*100:.2f}% "
            f"train Acc={lin_mets['train_avg_acc']*100:.2f}%±{lin_mets['train_std_acc']*100:.2f}% | "
            f"val F1={lin_mets['val_avg_f1']*100:.2f}%±{lin_mets['val_std_f1']*100:.2f}% "
            f"val Acc={lin_mets['val_avg_acc']*100:.2f}%±{lin_mets['val_std_acc']*100:.2f}% "
            f"(avg 1-F1 err {best_err:.2f}%)"
        )
        metrics_summary[name]['SVM_Linear'] = lin_mets

        # 2b. Custom SVM (SMO) using same C grid, averaged across folds (optional)
        if enable_custom_svm:
            print(f"  Custom SVM grid: {len(svm_C_list_linear)} C values (averaging over folds)")
            CsP2 = np.log2(svm_C_list_linear)
            # Accumulate per-C errors across folds
            per_c_err_sums = np.zeros(len(CsP2), dtype=float)
            per_c_counts = np.zeros(len(CsP2), dtype=int)
            start_ts = time.monotonic()
            fold_no = 0
            for train_idx, val_idx in skf.split(X_all, y_all):
                X_tr, X_va = X_all[train_idx], X_all[val_idx]
                y_tr, y_va = y_all[train_idx], y_all[val_idx]
                avgErrVec, stdErrVec, _optC = grid_search_svm_custom(
                    X_tr, y_tr, X_va, y_va, CsP2
                )
                per_c_err_sums += avgErrVec
                per_c_counts += 1
                fold_no += 1
                elapsed = time.monotonic() - start_ts
                print(f"    Custom SVM fold {fold_no}/{k} aggregated (elapsed {elapsed:.1f}s)")
                sys.stdout.flush()
            per_c_err_avgs = per_c_err_sums / np.maximum(per_c_counts, 1)
            best_c_idx = int(np.argmin(per_c_err_avgs))
            best_c_custom = svm_C_list_linear[best_c_idx]
            print(f"  -> Optimal C (Custom SVM): {best_c_custom} (avg error {per_c_err_avgs[best_c_idx]:.2f}%)")
            optimal_params[f'SVM_Custom_{name}'] = best_c_custom

        # 3. RBF SVM (C, gamma)
        best_err = np.inf
        best_c_rbf, best_g_rbf = None, None
        total_combos = len(svm_C_list_rbf) * len(svm_gamma_list)
        print(f"  RBF SVM grid: {len(svm_C_list_rbf)} C x {len(svm_gamma_list)} gamma = {total_combos} combos")
        combo_i = 0
        start_ts = time.monotonic()
        for C_idx, C in enumerate(svm_C_list_rbf, start=1):
            for gamma in svm_gamma_list:
                def _eval_svm_rbf(X_tr, y_tr, X_va, y_va, C_val=C, G_val=gamma):
                    clf = SVC(kernel="rbf", C=C_val, gamma=G_val, random_state=42)
                    clf.fit(X_tr, y_tr)
                    y_pred = clf.predict(X_va)
                    return 100 * (1 - f1_score(y_va, y_pred))
                avg_err, std_err = avg_error_over_folds(_eval_svm_rbf)
                combo_i += 1
                if combo_i % max(1, progress_every) == 0 or combo_i == 1:
                    pct = 100.0 * combo_i / total_combos
                    elapsed = time.monotonic() - start_ts
                    eta = (elapsed / max(combo_i, 1)) * (total_combos - combo_i)
                    print(f"    combo {combo_i}/{total_combos} ({pct:.1f}%) C={C:.4f} gamma={gamma:.6f} avg_err={avg_err:.2f}% (elapsed {elapsed:.1f}s, ETA {eta:.1f}s)")
                    sys.stdout.flush()
                if avg_err < best_err:
                    best_err, best_c_rbf, best_g_rbf = avg_err, C, gamma
        optimal_params[f'SVM_RBF_{name}'] = (best_c_rbf, best_g_rbf)
        # Print avg F1/Acc for best RBF
        def _build_svm_rbf(Xtr, ytr, C_val=best_c_rbf, G_val=best_g_rbf):
            clf = SVC(kernel="rbf", C=C_val, gamma=G_val, random_state=42)
            clf.fit(Xtr, ytr)
            return clf
        if best_c_rbf is not None and best_g_rbf is not None:
            rbf_mets = avg_train_val_metrics_over_folds(lambda Xtr, ytr: _build_svm_rbf(Xtr, ytr, best_c_rbf, best_g_rbf))
            print(
                f"  -> Optimal RBF: C={best_c_rbf}, gamma={best_g_rbf} | "
                f"train F1={rbf_mets['train_avg_f1']*100:.2f}%±{rbf_mets['train_std_f1']*100:.2f}% "
                f"train Acc={rbf_mets['train_avg_acc']*100:.2f}%±{rbf_mets['train_std_acc']*100:.2f}% | "
                f"val F1={rbf_mets['val_avg_f1']*100:.2f}%±{rbf_mets['val_std_f1']*100:.2f}% "
                f"val Acc={rbf_mets['val_avg_acc']*100:.2f}%±{rbf_mets['val_std_acc']*100:.2f}% "
                f"(avg 1-F1 err {best_err:.2f}%)"
            )
            metrics_summary[name]['SVM_RBF'] = rbf_mets
        else:
            print(f"  -> Optimal RBF: C={best_c_rbf}, gamma={best_g_rbf} (avg 1-F1 err {best_err:.2f}%)")

        # For DT/AdaBoost/RF, operate on X matrices directly per fold
        # 4. Decision Tree
        best_err = np.inf
        best_dt = {"max_depth": None, "criterion": None}
        total_dt = len(dt_depth_list) * len(dt_criterion_list)
        print(f"  Decision Tree grid: {len(dt_depth_list)} depths x {len(dt_criterion_list)} criteria = {total_dt} combos")
        dt_i = 0
        start_ts = time.monotonic()
        for depth in dt_depth_list:
            for crit in dt_criterion_list:
                def _eval_dt(X_tr, y_tr, X_va, y_va, d=depth, c=crit):
                    clf = DecisionTreeClassifier(max_depth=d, criterion=c, random_state=42)
                    clf.fit(X_tr, y_tr)
                    y_pred = clf.predict(X_va)
                    return 100 * (1 - f1_score(y_va, y_pred))
                avg_err, std_err = avg_error_over_folds(_eval_dt)
                dt_i += 1
                if dt_i % max(1, progress_every//2) == 0 or dt_i == 1:
                    pct = 100.0 * dt_i / total_dt
                    elapsed = time.monotonic() - start_ts
                    eta = (elapsed / max(dt_i, 1)) * (total_dt - dt_i)
                    print(f"    DT combo {dt_i}/{total_dt} ({pct:.1f}%) depth={depth} crit={crit} avg_err={avg_err:.2f}% (elapsed {elapsed:.1f}s, ETA {eta:.1f}s)")
                    sys.stdout.flush()
                if avg_err < best_err:
                    best_err = avg_err
                    best_dt = {"max_depth": depth, "criterion": crit}
        optimal_params[f'DT_{name}'] = best_dt
        # Print avg F1/Acc for best DT
        def _build_dt(Xtr, ytr, depth=best_dt['max_depth'], crit=best_dt['criterion']):
            clf = DecisionTreeClassifier(max_depth=depth, criterion=crit, random_state=42)
            clf.fit(Xtr, ytr)
            return clf
        dt_mets = avg_train_val_metrics_over_folds(lambda Xtr, ytr: _build_dt(Xtr, ytr, best_dt['max_depth'], best_dt['criterion']))
        print(
            f"  -> Optimal DT: {best_dt} | "
            f"train F1={dt_mets['train_avg_f1']*100:.2f}%±{dt_mets['train_std_f1']*100:.2f}% "
            f"train Acc={dt_mets['train_avg_acc']*100:.2f}%±{dt_mets['train_std_acc']*100:.2f}% | "
            f"val F1={dt_mets['val_avg_f1']*100:.2f}%±{dt_mets['val_std_f1']*100:.2f}% "
            f"val Acc={dt_mets['val_avg_acc']*100:.2f}%±{dt_mets['val_std_acc']*100:.2f}% "
            f"(avg 1-F1 err {best_err:.2f}%)"
        )
        metrics_summary[name]['DT'] = dt_mets

        # 5. AdaBoost (sklearn)
        best_err = np.inf
        best_ab = {"n_estimators": None, "learning_rate": None}
        total_ab = len(ab_n_est_list) * len(ab_lr_list)
        print(f"  AdaBoost grid: {len(ab_n_est_list)} n_estimators x {len(ab_lr_list)} learning_rates = {total_ab} combos")
        ab_i = 0
        start_ts = time.monotonic()
        for n_est in ab_n_est_list:
            for lr in ab_lr_list:
                def _eval_ab(X_tr, y_tr, X_va, y_va, n=n_est, r=lr):
                    clf = AdaBoostClassifier(n_estimators=n, learning_rate=r, random_state=42)
                    clf.fit(X_tr, y_tr)
                    y_pred = clf.predict(X_va)
                    return 100 * (1 - f1_score(y_va, y_pred))
                avg_err, std_err = avg_error_over_folds(_eval_ab)
                ab_i += 1
                if ab_i % max(1, progress_every) == 0 or ab_i == 1:
                    pct = 100.0 * ab_i / total_ab
                    elapsed = time.monotonic() - start_ts
                    eta = (elapsed / max(ab_i, 1)) * (total_ab - ab_i)
                    print(f"    AB combo {ab_i}/{total_ab} ({pct:.1f}%) n={n_est} lr={lr} avg_err={avg_err:.2f}% (elapsed {elapsed:.1f}s, ETA {eta:.1f}s)")
                    sys.stdout.flush()
                if avg_err < best_err:
                    best_err = avg_err
                    best_ab = {"n_estimators": n_est, "learning_rate": lr}
        optimal_params[f'AB_Sklearn_{name}'] = best_ab
        # Print avg F1/Acc for best AdaBoost
        def _build_ab(Xtr, ytr, n=best_ab['n_estimators'], r=best_ab['learning_rate']):
            clf = AdaBoostClassifier(n_estimators=n, learning_rate=r, random_state=42)
            clf.fit(Xtr, ytr)
            return clf
        ab_mets = avg_train_val_metrics_over_folds(lambda Xtr, ytr: _build_ab(Xtr, ytr, best_ab['n_estimators'], best_ab['learning_rate']))
        print(
            f"  -> Optimal AB: {best_ab} | "
            f"train F1={ab_mets['train_avg_f1']*100:.2f}%±{ab_mets['train_std_f1']*100:.2f}% "
            f"train Acc={ab_mets['train_avg_acc']*100:.2f}%±{ab_mets['train_std_acc']*100:.2f}% | "
            f"val F1={ab_mets['val_avg_f1']*100:.2f}%±{ab_mets['val_std_f1']*100:.2f}% "
            f"val Acc={ab_mets['val_avg_acc']*100:.2f}%±{ab_mets['val_std_acc']*100:.2f}% "
            f"(avg 1-F1 err {best_err:.2f}%)"
        )
        metrics_summary[name]['AB_Sklearn'] = ab_mets

        # 6. Random Forest
        best_err = np.inf
        best_rf = {"n_estimators": None, "max_depth": None}
        total_rf = len(rf_n_est_list) * len(rf_max_depth_list)
        print(f"  Random Forest grid: {len(rf_n_est_list)} n_estimators x {len(rf_max_depth_list)} depths = {total_rf} combos")
        rf_i = 0
        start_ts = time.monotonic()
        for n_est in rf_n_est_list:
            for mdepth in rf_max_depth_list:
                def _eval_rf(X_tr, y_tr, X_va, y_va, n=n_est, d=mdepth):
                    # actual RF
                    from sklearn.ensemble import RandomForestClassifier
                    rf = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=42, class_weight='balanced')
                    rf.fit(X_tr, y_tr)
                    y_pred = rf.predict(X_va)
                    return 100 * (1 - f1_score(y_va, y_pred))
                avg_err, std_err = avg_error_over_folds(_eval_rf)
                rf_i += 1
                if rf_i % max(1, progress_every//2) == 0 or rf_i == 1:
                    pct = 100.0 * rf_i / total_rf
                    elapsed = time.monotonic() - start_ts
                    eta = (elapsed / max(rf_i, 1)) * (total_rf - rf_i)
                    print(f"    RF combo {rf_i}/{total_rf} ({pct:.1f}%) n={n_est} depth={mdepth} avg_err={avg_err:.2f}% (elapsed {elapsed:.1f}s, ETA {eta:.1f}s)")
                    sys.stdout.flush()
                if avg_err < best_err:
                    best_err = avg_err
                    best_rf = {"n_estimators": n_est, "max_depth": mdepth}
        optimal_params[f'RF_{name}'] = best_rf
        # Print avg F1/Acc for best RF
        def _build_rf(Xtr, ytr, n=best_rf['n_estimators'], d=best_rf['max_depth']):
            from sklearn.ensemble import RandomForestClassifier
            rf = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=42, class_weight='balanced')
            rf.fit(Xtr, ytr)
            return rf
        rf_mets = avg_train_val_metrics_over_folds(lambda Xtr, ytr: _build_rf(Xtr, ytr, best_rf['n_estimators'], best_rf['max_depth']))
        print(
            f"  -> Optimal RF: {best_rf} | "
            f"train F1={rf_mets['train_avg_f1']*100:.2f}%±{rf_mets['train_std_f1']*100:.2f}% "
            f"train Acc={rf_mets['train_avg_acc']*100:.2f}%±{rf_mets['train_std_acc']*100:.2f}% | "
            f"val F1={rf_mets['val_avg_f1']*100:.2f}%±{rf_mets['val_std_f1']*100:.2f}% "
            f"val Acc={rf_mets['val_avg_acc']*100:.2f}%±{rf_mets['val_std_acc']*100:.2f}% "
            f"(avg 1-F1 err {best_err:.2f}%)"
        )
        metrics_summary[name]['RF'] = rf_mets

    print("\n\n--- OPTIMAL PARAMETERS SUMMARY (K-Fold) ---")
    for key, val in optimal_params.items():
        print(f"  {key}: {val}")
    print("-" * 60)
    print("Note: Selection now optimizes F1 (not Accuracy).")

    print("\n--- METRICS SUMMARY (train/val avg±std over folds) ---")
    for feat_set, methods in metrics_summary.items():
        print(f"[{feat_set}]")
        for method, m in methods.items():
            print(
                f"  {method}: "
                f"train F1={m['train_avg_f1']*100:.2f}%±{m['train_std_f1']*100:.2f}% | "
                f"train Acc={m['train_avg_acc']*100:.2f}%±{m['train_std_acc']*100:.2f}% | "
                f"val F1={m['val_avg_f1']*100:.2f}%±{m['val_std_f1']*100:.2f}% | "
                f"val Acc={m['val_avg_acc']*100:.2f}%±{m['val_std_acc']*100:.2f}%"
            )

    return optimal_params, metrics_summary