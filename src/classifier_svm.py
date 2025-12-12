
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
import numpy as np
from sklearn import svm



# Helper function to convert 0/1 labels to -1/+1
def map_labels_to_pm1(y):
    # Converts 0 (Healthy) -> -1 and 1 (Cancer) -> 1
    return np.where(y == 1, 1, -1)

# Helper function to convert -1/+1 labels back to 0/1
def map_labels_to_01(y):
    # Converts -1 -> 0 and 1 -> 1
    return np.where(y == 1, 1, 0)


def train_svm_custom(feature_names, method_name="Unknown", data=None, ixHealthy=None, ixCancer=None, C=1, tol=1e-3, max_passes=10):
    """
    Trains a Linear SVM using the provided custom SMO implementation (svm_hard_margin logic).
    C is set to a large value (1e10) for hard-margin, matching the default in the original code.
    """
    if ixHealthy is None or ixCancer is None:
        ixHealthy = np.where(data["Classification"] == "Healthy")
        ixCancer = np.where(data["Classification"] == "Cancer")

    print(f"\n=== CUSTOM LINEAR SVM CLASSIFIER (C={C}, {method_name}) ===")
    print(f"Using features: {feature_names}")

    # --- Data Extraction and Label Conversion ---
    X_train = data[feature_names].iloc[ixHealthy[0]].values # Healthy
    X_train = np.concatenate([X_train, data[feature_names].iloc[ixCancer[0]].values], axis=0)
    
    # y_train: 0/1 labels from train indices
    y_train_01 = np.concatenate([np.zeros(len(ixHealthy[0])), np.ones(len(ixCancer[0]))])
    # Convert to -1/+1 for the SMO algorithm
    y_train_pm1 = map_labels_to_pm1(y_train_01) 


    # --------------------------------------------------------------------------------------
    # START: Code adapted directly from svm_hard_margin (with label change: y_train_pm1)
    
    X = np.asmatrix(X_train)
    y = np.asmatrix(y_train_pm1).T
    m, n = X.shape

    # Inner helper functions for SMO (keeping them outside to avoid copy-pasting their body)
    # NOTE: You must ensure clipAlphasJ and selectJrandom are available in the scope 
    # (e.g., defined outside both train/test functions or copied here).
    
    # Since they were defined *inside* the original function, we'll define them here:
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

    # Compute w
    w = np.zeros((1, n))
    y_vec = np.asarray(y).flatten()
    for i in range(m):
        w += float(alphas[i]) * y_vec[i] * np.asarray(X[i, :])
    w = np.asmatrix(w).T  # shape (n,1)

    # Support vectors
    #Identify samples that are support vectors
    ixSv = np.where(np.asarray(alphas).flatten() > 0)[0]


    Svs = np.asarray(X)[ixSv, :]
    ySvs = y_vec[ixSv]

    # Compute b via média de dois SVs (se existirem de ambos os lados)
    if np.any(ySvs == 1) and np.any(ySvs == -1):
        #Identify the positive and negative support vectors
        pos_sv = Svs[ySvs == 1][0]
        neg_sv = Svs[ySvs == -1][0]
        b_alt = -0.5 * (float(w.T @ np.asmatrix(pos_sv).T) + float(w.T @ np.asmatrix(neg_sv).T))
        b = b_alt
        
    # END: Code adapted from svm_hard_margin
    # --------------------------------------------------------------------------------------


    # Define the predict function based on the calculated w and b
    def predict_scores(Xte):
        # Xte must be a matrix or array suitable for matrix multiplication
        Xte = np.asmatrix(Xte) if isinstance(Xte, np.ndarray) else Xte
        return (Xte * w + b).A.flatten() # Scores/Distance to hyperplane

    def predict_labels(Xte):
        scores = predict_scores(Xte)
        # Prediction in {-1, 1}
        y_pred_pm1 = np.where(scores >= 0, 1, -1)
        # Convert back to {0, 1} for compatibility with run_all_classifiers
        return map_labels_to_01(y_pred_pm1)


    print(f"SVM weights (w): {w.A.flatten()}")
    print(f"SVM bias (b): {b}")

    # Store the essential parts and prediction functions
    model = {
        "feature_names": feature_names,
        "w": w,
        "b": b,
        "predict_labels": predict_labels,  # Returns 0/1 labels
        "predict_scores": predict_scores,  # Returns raw scores (distance to hyperplane)
    }

    # Print training metrics (custom SVM on training set)
    y_pred_tr = model["predict_labels"](np.asarray(X_train))
    decision_scores_tr = model["predict_scores"](np.asarray(X_train))
    # Convert original training labels back to 0/1
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
    """
    Trains a Linear Support Vector Machine (SVM) classifier.
    """
    if ixHealthy is None or ixCancer is None:
        ixHealthy = np.where(data["Classification"] == "Healthy")
        ixCancer = np.where(data["Classification"] == "Cancer")

    print(f"\n=== LINEAR SVM CLASSIFIER (C={C}, {method_name}) ===")
    print(f"Using features: {feature_names}")

    # Extract training data
    X_train = data[feature_names].values
    # Target labels must be 0/1 for consistency
    y_train = np.concatenate([np.zeros(len(ixHealthy[0])), np.ones(len(ixCancer[0]))]) 

    # Fit SVM model with linear kernel
    # probability=True allows predict_proba for ROC-AUC, though decision_function is often preferred for SVM scores.
    clf = svm.SVC(kernel="linear", C=C, random_state=42, probability=True) 
    clf.fit(X_train, y_train)

    # Note: SVM stores the decision boundary as w and b similar to your custom code, 
    # but we store the whole object for simplicity.
    w = clf.coef_.flatten()
    b = clf.intercept_[0]
    
    print(f"SVM weights: {w}")
    print(f"SVM bias: {b}")

    model = {
        "feature_names": feature_names,
        "clf": clf,
        "C": C,
    }

    # Print training metrics
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
    """
    Trains an RBF Kernel Support Vector Machine (SVM) classifier using pre-determined optimal C and gamma.
    """
    if ixHealthy is None or ixCancer is None:
        ixHealthy = np.where(data["Classification"] == "Healthy")
        ixCancer = np.where(data["Classification"] == "Cancer")

    print(f"\n=== RBF-KERNEL SVM CLASSIFIER (C={C:.4f}, Gamma={gamma:.6f}, {method_name}) ===")
    print(f"Using features: {feature_names}")

    # Extract training data
    X_train = data[feature_names].values
    y_train = np.concatenate([np.zeros(len(ixHealthy[0])), np.ones(len(ixCancer[0]))]) # 0=Healthy, 1=Cancer

    # Fit RBF SVM model
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

    # Print training metrics
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
    """
    Classifies test data using either the Scikit-learn SVC/LinearSVC or the custom SVM model, 
    and computes metrics.
    """
    if ixHealthy is None or ixCancer is None:
        ixHealthy = np.where(data["Classification"] == "Healthy")
        ixCancer = np.where(data["Classification"] == "Cancer")

    feature_names = model["feature_names"]

    # --- Test Data Preparation ---
    # Need to handle both DataFrame slice and full array/matrix input for compatibility
    if "w" in model: # Custom model uses array/matrix multiplication and works best with matrix data
        X_test = data[feature_names].iloc[ixHealthy[0]].values
        X_test = np.concatenate([X_test, data[feature_names].iloc[ixCancer[0]].values], axis=0)
    else: # Scikit-learn model handles the dataframe slice extraction within the predict method
        X_test = data[feature_names].values
        
    y_true = np.concatenate([np.zeros(len(ixHealthy[0])), np.ones(len(ixCancer[0]))])

    # --- Prediction and Decision Scores (Unified Logic) ---
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


    # --- Metric Calculation ---
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



