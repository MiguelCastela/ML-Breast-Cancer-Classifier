
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
import numpy as np
from sklearn import svm



# Helper function to convert 0/1 labels to -1/+1
def map_labels_to_pm1(y):
    return np.where(y == 1, 1, -1)

# Helper function to convert -1/+1 labels back to 0/1
def map_labels_to_01(y):
    return np.where(y == 1, 1, 0)


def train_svm_custom(feature_names, method_name="Unknown", data=None, ixHealthy=None, ixCancer=None, C=1, tol=1e-3, max_passes=10):
    
    if ixHealthy is None or ixCancer is None:
        ixHealthy = np.where(data["Classification"] == "Healthy")
        ixCancer = np.where(data["Classification"] == "Cancer")

    print(f"\n=== CUSTOM LINEAR SVM CLASSIFIER (C={C}, {method_name}) ===")
    print(f"Using features: {feature_names}")

    X_train = data[feature_names].iloc[ixHealthy[0]].values # Healthy
    X_train = np.concatenate([X_train, data[feature_names].iloc[ixCancer[0]].values], axis=0)
    
    y_train_01 = np.concatenate([np.zeros(len(ixHealthy[0])), np.ones(len(ixCancer[0]))])
    y_train_pm1 = map_labels_to_pm1(y_train_01) 


    
    X = np.asmatrix(X_train)
    y = np.asmatrix(y_train_pm1).T
    m, n = X.shape

    # Inner helper functions for SMO (keeping them outside to avoid copy-pasting their body)
    # NOTE: You must ensure clipAlphasJ and selectJrandom are available in the scope 
    # (e.g., defined outside both train/test functions or copied here).
    
    def clipAlphasJ(aj, H, L):
        if aj > H: aj = H
        if L > aj: aj = L
        return aj

    def selectJrandom(i, m):
        j = i
        while j == i:
            j = int(np.random.uniform(0, m))
        return j


    b = 0.0
    alphas = np.asmatrix(np.zeros((m, 1)))
    passes = 0

    while passes < max_passes:
        num_changed_alphas = 0
        for i in range(m):
            # The calculation of fXi uses the *current* alphas and b
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


    ixSv = np.where(np.asarray(alphas).flatten() > 0)[0]


    Svs = np.asarray(X)[ixSv, :]
    ySvs = y_vec[ixSv]

    if np.any(ySvs == 1) and np.any(ySvs == -1):
        pos_sv = Svs[ySvs == 1][0]
        neg_sv = Svs[ySvs == -1][0]
        b_alt = -0.5 * (float(w.T @ np.asmatrix(pos_sv).T) + float(w.T @ np.asmatrix(neg_sv).T))
        b = b_alt
        


    def predict_scores(Xte):
        Xte = np.asmatrix(Xte) if isinstance(Xte, np.ndarray) else Xte
        return (Xte * w + b).A.flatten() 

    def predict_labels(Xte):
        scores = predict_scores(Xte)
        y_pred_pm1 = np.where(scores >= 0, 1, -1)
        return map_labels_to_01(y_pred_pm1)


    print(f"SVM weights (w): {w.A.flatten()}")
    print(f"SVM bias (b): {b}")

    model = {
        "feature_names": feature_names,
        "w": w,
        "b": b,
        "predict_labels": predict_labels,  
        "predict_scores": predict_scores,  
    }

    y_pred_tr = model["predict_labels"](np.asarray(X_train))
    decision_scores_tr = model["predict_scores"](np.asarray(X_train))
    y_train = np.where(y_train_pm1 == 1, 1, 0)
    TP = np.sum((y_train == 1) & (y_pred_tr == 1))
    TN = np.sum((y_train == 0) & (y_pred_tr == 0))
    FP = np.sum((y_train == 0) & (y_pred_tr == 1))
    FN = np.sum((y_train == 1) & (y_pred_tr == 0))
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_train, decision_scores_tr)
    roc_auc = auc(fpr, tpr)
    print(f"[Train] Sensitivity (%) = {sensitivity * 100:.2f}")
    print(f"[Train] Specificity (%) = {specificity * 100:.2f}")
    print(f"[Train] Precision (%) = {precision * 100:.2f}")
    print(f"[Train] F1 Score (%) = {f1 * 100:.2f}")
    print(f"[Train] Accuracy (%) = {accuracy * 100:.2f}")
    print(f"[Train] ROC-AUC (%) = {roc_auc * 100:.2f}")

    return (
        model,
        accuracy,
        sensitivity,
        specificity,
        precision,
        f1,
        roc_auc,
    )



def train_svm_linear(feature_names, method_name="Unknown", data=None, ixHealthy=None, ixCancer=None, C=1):
    
    if ixHealthy is None or ixCancer is None:
        ixHealthy = np.where(data["Classification"] == "Healthy")
        ixCancer = np.where(data["Classification"] == "Cancer")

    print(f"\n=== LINEAR SVM CLASSIFIER (C={C}, {method_name}) ===")
    print(f"Using features: {feature_names}")

    X_train = data[feature_names].values
    y_train = np.concatenate([np.zeros(len(ixHealthy[0])), np.ones(len(ixCancer[0]))]) 

    
    clf = svm.SVC(kernel="linear", C=C, random_state=42, probability=True) 
    clf.fit(X_train, y_train)

    
    w = clf.coef_.flatten()
    b = clf.intercept_[0]
    
    print(f"SVM weights: {w}")
    print(f"SVM bias: {b}")

    model = {
        "feature_names": feature_names,
        "clf": clf,
        "C": C,
    }

    y_pred_tr = clf.predict(X_train)
    decision_scores_tr = clf.predict_proba(X_train)[:, 1]
    TP = np.sum((y_train == 1) & (y_pred_tr == 1))
    TN = np.sum((y_train == 0) & (y_pred_tr == 0))
    FP = np.sum((y_train == 0) & (y_pred_tr == 1))
    FN = np.sum((y_train == 1) & (y_pred_tr == 0))
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_train, decision_scores_tr)
    roc_auc = auc(fpr, tpr)
    print(f"[Train] Sensitivity (%) = {sensitivity * 100:.2f}")
    print(f"[Train] Specificity (%) = {specificity * 100:.2f}")
    print(f"[Train] Precision (%) = {precision * 100:.2f}")
    print(f"[Train] F1 Score (%) = {f1 * 100:.2f}")
    print(f"[Train] Accuracy (%) = {accuracy * 100:.2f}")
    print(f"[Train] ROC-AUC (%) = {roc_auc * 100:.2f}")

    return (
        model,
        accuracy,
        sensitivity,
        specificity,
        precision,
        f1,
        roc_auc,
    )





def train_svm_rbf_kernel(feature_names, method_name="Unknown", data=None, ixHealthy=None, ixCancer=None, C=2**5.0, gamma=2**-15.0):
    
    if ixHealthy is None or ixCancer is None:
        ixHealthy = np.where(data["Classification"] == "Healthy")
        ixCancer = np.where(data["Classification"] == "Cancer")

    print(f"\n=== RBF-KERNEL SVM CLASSIFIER (C={C:.4f}, Gamma={gamma:.6f}, {method_name}) ===")
    print(f"Using features: {feature_names}")

    X_train = data[feature_names].values
    y_train = np.concatenate([np.zeros(len(ixHealthy[0])), np.ones(len(ixCancer[0]))]) # 0=Healthy, 1=Cancer

    
    # probability=True is necessary to get predict_proba for ROC-AUC if needed, 
    # although decision_function is generally used for scores.
    clf = svm.SVC(kernel="rbf", C=C, gamma=gamma, random_state=42, probability=True) 
    clf.fit(X_train, y_train)

    model = {
        "feature_names": feature_names,
        "clf": clf,
        "C": C,
        "gamma": gamma
    }

    y_pred_tr = clf.predict(X_train)
    decision_scores_tr = clf.predict_proba(X_train)[:, 1]
    TP = np.sum((y_train == 1) & (y_pred_tr == 1))
    TN = np.sum((y_train == 0) & (y_pred_tr == 0))
    FP = np.sum((y_train == 0) & (y_pred_tr == 1))
    FN = np.sum((y_train == 1) & (y_pred_tr == 0))
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_train, decision_scores_tr)
    roc_auc = auc(fpr, tpr)
    print(f"[Train] Sensitivity (%) = {sensitivity * 100:.2f}")
    print(f"[Train] Specificity (%) = {specificity * 100:.2f}")
    print(f"[Train] Precision (%) = {precision * 100:.2f}")
    print(f"[Train] F1 Score (%) = {f1 * 100:.2f}")
    print(f"[Train] Accuracy (%) = {accuracy * 100:.2f}")
    print(f"[Train] ROC-AUC (%) = {roc_auc * 100:.2f}")

    return (
        model,
        accuracy,
        sensitivity,
        specificity,
        precision,
        f1,
        roc_auc,
    )



def test_svm_generic(model, data=None , ixHealthy=None, ixCancer=None):
    
    if ixHealthy is None or ixCancer is None:
        ixHealthy = np.where(data["Classification"] == "Healthy")
        ixCancer = np.where(data["Classification"] == "Cancer")

    feature_names = model["feature_names"]

    
    # Need to handle both DataFrame slice and full array/matrix input for compatibility
    if "w" in model: # Custom model uses array/matrix multiplication and works best with matrix data
        X_test = data[feature_names].iloc[ixHealthy[0]].values
        X_test = np.concatenate([X_test, data[feature_names].iloc[ixCancer[0]].values], axis=0)
    else: # Scikit-learn model handles the dataframe slice extraction within the predict method
        X_test = data[feature_names].values
        
    y_true = np.concatenate([np.zeros(len(ixHealthy[0])), np.ones(len(ixCancer[0]))])

    if "clf" in model:
        # Case 1: Scikit-learn model (Trained by train_svm_linear or train_svm_rbf)
        clf = model["clf"]
        y_pred = clf.predict(X_test)
        # Decision function for Scikit-learn SVMs
        decision_scores = clf.decision_function(X_test)
    
    elif "predict_labels" in model:
        # Case 2: Custom model (Trained by train_svm_custom)
        y_pred = model["predict_labels"](X_test)
        decision_scores = model["predict_scores"](X_test)
    
    else:
        raise ValueError("Model dictionary must contain either 'clf' (sklearn) or 'predict_labels' (custom).")


    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    fpr, tpr, _ = roc_curve(y_true, decision_scores)
    roc_auc = auc(fpr, tpr)

    print(f"Sensitivity (%) = {sensitivity * 100:.2f}")
    print(f"Specificity (%) = {specificity * 100:.2f}")
    print(f"Precision (%) = {precision * 100:.2f}")
    print(f"F1 Score (%) = {f1_score * 100:.2f}")
    print(f"Accuracy (%) = {accuracy * 100:.2f}")
    print(f"ROC-AUC (%) = {roc_auc * 100:.2f}")

    return accuracy, sensitivity, specificity, precision, f1_score, roc_auc



