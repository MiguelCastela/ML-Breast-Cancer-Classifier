
import numpy as np
import plotly.express as px
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from classifier_decision_tree import test_decision_tree


def grid_search_svm_linear(X_train, y_train, X_val, y_val, CsP2):
    
    # Single-run error vector (1 - F1) * 100
    errVec = np.zeros(CsP2.shape[0])
    ki = 0
    print(f"\nProcessing Linear SVM grid on single split...")
    
    for cP2 in CsP2: 
        C = 2**cP2
        
        clf = svm.SVC(kernel="linear", C=C, random_state=42)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_val)
        
        f1 = f1_score(y_val, y_pred)
        
        errVec[ki] = (1 - f1) * 100
        
        print(f"  C = {C:<10.4f} (2^{cP2}): Validation F1 = {f1 * 100:.2f}% | Error = {errVec[ki]:.2f}%")
        
        ki = ki + 1

    # Single split: avgError is errVec; stdError zeros
    avgError = errVec
    stdError = np.zeros_like(avgError)
    # Map to actual C list for selection
    C_list = 2**CsP2
    optC = C_list[np.argmin(avgError)]
    return avgError, stdError, optC


def grid_search_svm_rbf(X_train, y_train, X_val, y_val, CsP2, GammasP2):
    
    
    errMat = np.zeros((CsP2.shape[0], GammasP2.shape[0]))
    
    best_error = np.inf
    optC, optGamma = 0, 0
    
    print(f"\n=== Starting RBF-SVM Grid Search ({CsP2.shape[0]} C values x {GammasP2.shape[0]} Gamma values) ===")
    
    ci = 0
    for cP2 in CsP2: 
        C = 2**cP2
        
        gi = 0
        for gP2 in GammasP2: 
            gamma = 2**gP2
            
            clf = svm.SVC(kernel="rbf", C=C, gamma=gamma, random_state=42)
            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_val)
            
            f1 = f1_score(y_val, y_pred)
            error_rate = (1 - f1) * 100
            
            errMat[ci, gi] = error_rate
            
            if error_rate < best_error:
                best_error = error_rate
                optC = C
                optGamma = gamma
            
            print(f"  C={C:<8.4f} | Gamma={gamma:<10.6f} | F1={f1 * 100:.2f}% | Error={error_rate:.2f}%")
            
            gi += 1
        ci += 1

    avgError = errMat
    stdError = np.zeros_like(avgError)
    
    print(f"\n--- Grid Search Complete ---")
    print(f"Optimal Parameters (F1-based): C={optC:.4f}, Gamma={optGamma:.6f} (Min Error: {best_error:.2f}%)")

    return avgError, stdError, optC, optGamma



def grid_search_svm_custom(X_train, y_train, X_val, y_val, CsP2):

    def train_and_predict_custom_svm(X_train, y_train_01, C=1, tol=1e-3, max_passes=10):

       
            # Helper functions from classifier_svm.py
        def map_labels_to_pm1(y):
            # Converts 0 (Healthy) -> -1 and 1 (Cancer) -> 1
            return np.where(y == 1, 1, -1)

        def map_labels_to_01(y):
            # Converts -1 -> 0 and 1 -> 1
            return np.where(y == 1, 1, 0)

        def clipAlphasJ(aj, H, L):
            if aj > H: aj = H
            if L > aj: L = aj
            return aj

        def selectJrandom(i, m):
            j = i
            while j == i:
                j = int(np.random.uniform(0, m))
            return j

        y_train_pm1 = map_labels_to_pm1(y_train_01) 
        
        X = np.asmatrix(X_train)
        y = np.asmatrix(y_train_pm1).T
        m, n = X.shape

        b = 0.0
        alphas = np.asmatrix(np.zeros((m, 1)))
        passes = 0

        while passes < max_passes:
            num_changed_alphas = 0
            for i in range(m):
                fXi = float(np.multiply(alphas, y).T * (X * X[i, :].T)) + b
                Ei = fXi - float(y[i])
                if ((y[i] * Ei < -tol) and (alphas[i] < C)) or ((y[i] * Ei > tol) and (alphas[i] > 0)):
                    j = selectJrandom(i, m)
                    fXj = float(np.multiply(alphas, y).T * (X * X[j, :].T)) + b
                    Ej = fXj - float(y[j])

                    alphaIold = alphas[i].copy()
                    alphaJold = alphas[j].copy()

                    if (y[i] != y[j]):
                        L = max(0, alphas[j] - alphas[i])
                        H = min(C, C + alphas[j] - alphas[i])
                    else:
                        L = max(0, alphas[j] + alphas[i] - C)
                        H = min(C, alphas[j] + alphas[i])

                    if L == H:
                        continue

                    eta = 2.0 * X[i, :] * X[j, :].T - X[i, :] * X[i, :].T - X[j, :] * X[j, :].T
                    if eta >= 0:
                        continue

                    alphas[j] -= y[j] * (Ei - Ej) / eta
                    alphas[j] = clipAlphasJ(alphas[j], H, L)

                    if abs(alphas[j] - alphaJold) < 1e-5:
                        continue

                    alphas[i] += y[j] * y[i] * (alphaJold - alphas[j])

                    b1 = b - Ei - y[i] * (alphas[i] - alphaIold) * X[i, :] * X[i, :].T - y[j] * (alphas[j] - alphaJold) * X[i, :] * X[j, :].T
                    b2 = b - Ej - y[i] * (alphas[i] - alphaIold) * X[i, :] * X[j, :].T - y[j] * (alphas[j] - alphaJold) * X[j, :] * X[j, :].T

                    if (0 < alphas[i]) and (C > alphas[i]):
                        b = float(b1)
                    elif (0 < alphas[j]) and (C > alphas[j]):
                        b = float(b2)
                    else:
                        b = float((b1 + b2) / 2.0)

                    num_changed_alphas += 1

            passes = passes + 1 if num_changed_alphas == 0 else 0

        w = np.zeros((1, n))
        y_vec = np.asarray(y).flatten()
        for i in range(m):
            w += float(alphas[i]) * y_vec[i] * np.asarray(X[i, :])
        w = np.asmatrix(w).T  

        # Re-compute b via average of two SVs (optional, for stability, matching original)
        ixSv = np.where(np.asarray(alphas).flatten() > 0)[0]
        Svs = np.asarray(X)[ixSv, :]
        ySvs = y_vec[ixSv]
        if np.any(ySvs == 1) and np.any(ySvs == -1):
            pos_sv = Svs[ySvs == 1][0]
            neg_sv = Svs[ySvs == -1][0]
            b_alt = -0.5 * (float(w.T @ np.asmatrix(pos_sv).T) + float(w.T @ np.asmatrix(neg_sv).T))
            b = b_alt

        # Define the predict function based on the calculated w and b
        def predict_labels(Xte):
            Xte = np.asmatrix(Xte) if isinstance(Xte, np.ndarray) else Xte
            scores = (Xte * w + b).A.flatten() 
            y_pred_pm1 = np.where(scores >= 0, 1, -1)
            # Convert back to {0, 1} for compatibility with grid search
            return map_labels_to_01(y_pred_pm1)

        return predict_labels


    errMat = np.zeros((1, CsP2.shape[0]))
    
    # Store the last partition for final testing outside the loop (for structural similarity)
    global X_train_final, X_test_final, y_train_final, y_test_final
    X_train_final, y_train_final = X_train, y_train
    # X_val is treated as X_test/validation for this run
    X_test_final, y_test_final = X_val, y_val

    
    r = 0 
    ki = 0
    
    print(f"\n--- Running Custom Grid Search on Single Split ---")

    for cP2 in CsP2: 
        C_val = 2**cP2
        

        predict_labels = train_and_predict_custom_svm(X_train, y_train, C=C_val)
        
        y_pred = predict_labels(X_val)
        
        f1 = f1_score(y_val, y_pred)
        errMat[r, ki] = (1 - f1) * 100
        
        print(f"  C = {C_val:<10.4f} (2^{cP2}): Validation F1 = {f1 * 100:.2f}% | Error = {errMat[r, ki]:.2f}%")

        ki = ki + 1
            
    
    avgError = errMat[0, :] 
    stdError = np.zeros_like(avgError)
    C_list = 2**CsP2
    optC = C_list[np.argmin(avgError)]
    
    return avgError, stdError, optC

def grid_search_knn(X_train, y_train, X_val, y_val, ks):
    
    # errMat is now 1 row (single run) x number of k values
    errMat = np.zeros((1, ks.shape[0]))
    
    # Store the last partition for final testing outside the loop (for structural similarity)
    global X_train_final, X_test_final, y_train_final, y_test_final
    X_train_final, y_train_final = X_train, y_train
    X_test_final, y_test_final = X_val, y_val

    
    # --- Standardization (Still essential, performed once per call) ---
    # Obtain training standardization factors
    muTr = np.mean(X_train, axis=0)
    stdTr = np.std(X_train, axis=0)
    
    stdTr_safe = np.where(stdTr == 0, 1.0, stdTr) 

    X_train_std = (X_train - muTr) / stdTr_safe
    X_val_std = (X_val - muTr) / stdTr_safe 
    
    r = 0
    ki = 0
    
    print(f"\n--- Running KNN Grid Search on Single Split ---")

    for k in ks: 
        
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_std, y_train)
        
        yp = knn.predict(X_val_std)
        
        f1 = f1_score(y_val, yp)
        
        errMat[r, ki] = (1 - f1) * 100
        
        print(f"  K = {k:<2}: Validation F1 = {f1 * 100:.2f}% | Error = {errMat[r, ki]:.2f}%")
        
        ki = ki + 1
            
    
    avgError = errMat[0, :] 
    stdError = np.zeros_like(avgError)
    optK = ks[np.where(avgError == np.min(avgError))[0]][0]
    
    return avgError, stdError, optK

def grid_search_dt(data_train, data_val, feature_names, depth_list, criterion_list):
    
    
    def train_decision_tree(feature_names, method_name="Unknown", data=None, ixHealthy=None, ixCancer=None, max_depth=5, criterion='gini'):
       
        if ixHealthy is None or ixCancer is None:
            ixHealthy = np.where(data["Classification"] == "Healthy")
            ixCancer = np.where(data["Classification"] == "Cancer")

        X_train = data[feature_names].values
        y_train = np.concatenate([np.zeros(len(ixHealthy[0])), np.ones(len(ixCancer[0]))]) 

        clf = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=42)
        clf.fit(X_train, y_train)

        model = {
            "feature_names": feature_names,
            "clf": clf,
            "max_depth": max_depth,
            "criterion": criterion
        }

        return model

    best_f1 = -1
    best_params = {}
    errMat = np.zeros((len(depth_list), len(criterion_list)))
    
    results = []
    
    # Pre-calculate indices for efficiency on the TRAINING data
    ixHealthy_train = np.where(data_train["Classification"] == "Healthy")
    ixCancer_train = np.where(data_train["Classification"] == "Cancer")

    # Pre-calculate indices for efficiency on the VALIDATION data
    ixHealthy_val = np.where(data_val["Classification"] == "Healthy")
    ixCancer_val = np.where(data_val["Classification"] == "Cancer")


    print("--- Starting Decision Tree Grid Search (Validation Run) ---")

    for depth in depth_list:
        for criterion in criterion_list:
            
            # --- 1. Training ---
            model = train_decision_tree(
                feature_names=feature_names,
                method_name="GridSearch",
                data=data_train,           
                ixHealthy=ixHealthy_train, 
                ixCancer=ixCancer_train,   
                max_depth=depth,
                criterion=criterion
            )

            # --- 2. Testing & Evaluation ---
            accuracy, sensitivity, specificity, precision, f1, roc_auc = test_decision_tree(
                model=model,
                data=data_val,             
                ixHealthy=ixHealthy_val,   
                ixCancer=ixCancer_val      
            )

            # --- 3. Store Results ---
            current_result = {
                'max_depth': depth,
                'criterion': criterion,
                'f1_score': f1,
                'accuracy': accuracy,
                'roc_auc': roc_auc
            }
            results.append(current_result)
            
            di = depth_list.index(depth)
            ci = criterion_list.index(criterion)
            errMat[di, ci] = (1 - f1) * 100

            # --- 4. Update Best Model ---
            if f1 > best_f1:
                best_f1 = f1
                best_params = {'max_depth': depth, 'criterion': criterion}
                
            print(f"Params: depth={depth}, criterion='{criterion}' | F1-Score: {f1 * 100:.2f}%")

    print("--- Grid Search Complete ---")
    print(f"Optimal Parameters (based on F1-Score on Validation Data): {best_params}")
    
    avgError = errMat
    stdError = np.zeros_like(avgError)
    optDepth = best_params['max_depth']
    optCriterion = best_params['criterion']
    
    return avgError, stdError, optDepth, optCriterion, pd.DataFrame(results)

def grid_search_adaboost(X_train, y_train, X_val, y_val, n_estimators_list, learning_rate_list):
    
    n_est_arr = np.array(n_estimators_list)
    lr_arr = np.array(learning_rate_list)
    
    # Error matrix dimensions: n_estimators x learning_rate (now 2D, as runs loop is gone)
    errMat = np.zeros((n_est_arr.shape[0], lr_arr.shape[0])) # 2D
    
    # Store the last partition for potential external use (matching previous functions' global usage)
    global X_train_final, X_test_final, y_train_final, y_test_final
    X_train_final, y_train_final = X_train, y_train
    # X_val replaces X_test for this run
    X_test_final, y_test_final = X_val, y_val 
    
    
    ni = 0
    best_error = np.inf
    best_f1, best_acc = 0.0, 0.0
    best_n, best_lr = None, None
    for n_est in n_est_arr: 
        li = 0
        for lr in lr_arr: 
            
            # 1. Define base estimator (Decision Stump)
            base_estimator = DecisionTreeClassifier(max_depth=1)
            
            # 2. Fit AdaBoost classifier (using X_train, y_train)
            clf = AdaBoostClassifier(
                estimator=base_estimator,
                n_estimators=n_est,
                learning_rate=lr,
                random_state=42,
                algorithm='SAMME'
            )
            clf.fit(X_train, y_train)
            
            # 3. Predict and Evaluate error (using X_val, y_val)
            yp = clf.predict(X_val) 
            # F1-based error and track accuracy
            f1 = f1_score(y_val, yp)
            acc = accuracy_score(y_val, yp)
            err = (1 - f1) * 100
            errMat[ni, li] = err
            if err < best_error:
                best_error = err
                best_f1 = f1
                best_acc = acc
                best_n = n_est
                best_lr = lr
            
            li += 1
        ni += 1
            
   
    avgError = errMat 
    stdError = np.zeros_like(avgError) 
    
    # Find optimal parameters
    min_error_idx = np.argmin(avgError)
    
    # Convert flat index back to 2D indices (ni, li)
    opt_ni, opt_li = np.unravel_index(min_error_idx, avgError.shape)
    
    optN = n_est_arr[opt_ni]
    optLR = lr_arr[opt_li]
    
    print(f"\nOptimal n_estimators (Validation Run): {optN}")
    print(f"Optimal learning_rate (Validation Run): {optLR}")
    print(f"Best F1: {best_f1*100:.2f}% | Best Acc: {best_acc*100:.2f}% | Min 1-F1 Err: {best_error:.2f}%")
    
    return avgError, stdError, optN, optLR, {"best_f1": best_f1, "best_acc": best_acc}


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def grid_search_adaboost_custom(data_train_df, data_val_df, feature_names, n_estimators_list, learning_rate_list):
    
    def map_labels_to_pm1(y_01):
        return np.where(y_01 == 1, 1, -1)
    
    def map_labels_to_01(y_pm1):
        return np.where(y_pm1 == 1, 1, 0)
    
    def trainAdaboost(Xtr, ytr, nMod, learning_rate):
        models = np.arange(nMod, dtype=DecisionTreeClassifier)
        N = Xtr.shape[0]
        w = np.ones((N)) * (1/N)
        
        for i in range(nMod):
            mdl = DecisionTreeClassifier(criterion='gini', max_depth=1) 
            mdl.fit(Xtr, ytr, sample_weight=w)
            pred = mdl.predict(Xtr)
            models[i] = mdl
            
            err = np.sum(w * np.heaviside(-ytr * pred, 1)) 
            
            if err < 1e-10: 
                err = 1e-10
            elif err >= 1.0 - 1e-10:
                err = 1.0 - 1e-10

            alpha_classic = 0.5 * np.log((1 - err) / err)
            alpha = learning_rate * alpha_classic
            models[i].alpha = alpha
            
            v = w * np.exp(-alpha_classic * ytr * pred)
            Sm = np.sum(v)
            w = v / Sm
            
        return models
    
    def testAdaboost(Xte, models):
        nMod = models.shape[0]
        nsamp = Xte.shape[0]
        predTot = np.zeros((nMod, nsamp))
        
        for m in range(nMod):
            label = models[m].predict(Xte) 
            predTot[m,:] = models[m].alpha * label 
            
        decision_scores = np.sum(predTot, 0)
        y_pred_pm1 = np.sign(decision_scores) 
        
        return y_pred_pm1, decision_scores


    # --- Grid Search Core Logic ---

    n_est_arr = np.array(n_estimators_list)
    lr_arr = np.array(learning_rate_list)
    
    # Error matrix dimensions: n_estimators x learning_rate (2D, as 'runs' is removed)
    errMat = np.zeros((n_est_arr.shape[0], lr_arr.shape[0])) # 2D

    # The internal 'runs' loop is REMOVED. The function now processes one run.
    r = 0 

    # --- Data Preparation (Replaces internal split logic) ---
    
    # Extract features and map labels (0/1 -> -1/+1) for training set
    X_train = data_train_df[feature_names].values
    y_train_01 = np.where(data_train_df["Classification"] == "Cancer", 1, 0)
    y_train_pm1 = map_labels_to_pm1(y_train_01) 

    X_val = data_val_df[feature_names].values
    y_val_true_01 = np.where(data_val_df["Classification"] == "Cancer", 1, 0) 


    ni = 0
    best_error = np.inf
    best_f1, best_acc = 0.0, 0.0
    best_n, best_lr = None, None
    for n_est in n_est_arr: 
        li = 0
        for lr in lr_arr: 
            
            # 1. Train the custom AdaBoost model
            trained_models = trainAdaboost(X_train, y_train_pm1, n_est, lr)
            
            # 2. Test the custom AdaBoost model (on X_val)
            y_pred_pm1, decision_scores = testAdaboost(X_val, trained_models) 
            
            # Final prediction must be converted back to 0/1 for metric calculation
            y_pred_01 = map_labels_to_01(y_pred_pm1)

            # 3. Calculate F1 and Accuracy
            f1 = f1_score(y_val_true_01, y_pred_01)
            acc = accuracy_score(y_val_true_01, y_pred_01)
            err = (1 - f1) * 100
            
            # 4. Store error (1 - F1) * 100 and track best
            errMat[ni, li] = err
            if err < best_error:
                best_error = err
                best_f1 = f1
                best_acc = acc
                best_n = n_est
                best_lr = lr
            
            li += 1
        ni += 1
            
   
    avgError = errMat 
    stdError = np.zeros_like(avgError) 
    
    # Find optimal parameters
    min_error_idx = np.argmin(avgError)
    
    # Convert flat index back to 2D indices (ni, li)
    opt_ni, opt_li = np.unravel_index(min_error_idx, avgError.shape)
    
    optN = n_est_arr[opt_ni]
    optLR = lr_arr[opt_li]
    
    print(f"\nOptimal n_estimators (n_mod): **{optN}**")
    print(f"Optimal learning_rate: **{optLR:.4f}**")
    print(f"Best F1: {best_f1*100:.2f}% | Best Acc: {best_acc*100:.2f}% | Min 1-F1 Err: {best_error:.2f}%")
    return avgError, stdError, optN, optLR, {"best_f1": best_f1, "best_acc": best_acc}

#oi olha é necessário isto aqui?? estes import outra vez??

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score 

def grid_search_random_forest(data_train, data_val, feature_names, n_estimators_list, max_depth_list):
    
    n_est_arr = np.array(n_estimators_list)
    depth_arr = np.array(max_depth_list)
    
    # Error matrix dimensions: n_estimators x max_depth
    errMat = np.zeros((n_est_arr.shape[0], depth_arr.shape[0])) 
    
   
    X_train = data_train[feature_names].values
    y_train = np.where(data_train["Classification"] == "Cancer", 1, 0)
    
    X_val = data_val[feature_names].values
    y_val = np.where(data_val["Classification"] == "Cancer", 1, 0)

    
    ni = 0
    best_error = np.inf
    optN, optD = 0, 0
    best_f1, best_acc = 0.0, 0.0

    print("\n--- Starting Random Forest Grid Search (Validation Run) ---")

    for n_est in n_est_arr: 
        di = 0
        for depth in depth_arr: 
            
            # 1. Fit Random Forest model (using X_train, y_train)
            clf = RandomForestClassifier(
                n_estimators=n_est,
                max_depth=depth if depth is not None else None,
                random_state=42,
            )
            clf.fit(X_train, y_train)
            
            # 2. Predict on validation data
            y_pred = clf.predict(X_val)
            
            # 3. Evaluate error via F1 (1 - F1) * 100 and track Accuracy
            f1 = f1_score(y_val, y_pred)
            acc = accuracy_score(y_val, y_pred)
            error_rate = (1 - f1) * 100
            
            # Store error
            errMat[ni, di] = error_rate
            
            depth_str = str(depth) if depth is not None else "None"
            print(f"Params: n_est={n_est}, depth={depth_str} | F1={f1 * 100:.2f}% Acc={acc*100:.2f}% | Err(1-F1)={error_rate:.2f}%")
            
            # 4. Update Optimal Parameters
            if error_rate < best_error:
                best_error = error_rate
                optN = n_est
                optD = depth
                best_f1 = f1
                best_acc = acc
            
            di += 1
        ni += 1
            
    
    avgError = errMat 
    stdError = np.zeros_like(avgError)
    
    print(f"\nOptimal n_estimators: **{optN}**")
    print(f"Optimal max_depth: **{optD}**")
    print(f"Best F1: **{best_f1*100:.2f}%** | Best Acc: **{best_acc*100:.2f}%** | Min 1-F1 Err: **{best_error:.2f}%**")

    return avgError, stdError, optN, optD, {"best_f1": best_f1, "best_acc": best_acc}