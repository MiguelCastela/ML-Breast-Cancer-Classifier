import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
import numpy as np


def train_fisher_lda(feature_names, method_name="Unknown", data=None , ixHealthy=None, ixCancer=None):

    if ixHealthy is None or ixCancer is None:
        ixHealthy = np.where(data["Classification"] == "Healthy")
        ixCancer = np.where(data["Classification"] == "Cancer")
    
    print(f"\n=== FISHER'S LDA CLASSIFIER ({method_name}) ===")
    print(f"Using features: {feature_names}")


    X_healthy = data[feature_names].iloc[ixHealthy[0]].values
    X_cancer = data[feature_names].iloc[ixCancer[0]].values
    
    X = np.concatenate([X_healthy, X_cancer], axis=0)
    y_true = np.concatenate([np.zeros(len(X_healthy)), np.ones(len(X_cancer))])  
    
    mu_healthy = np.mean(X_healthy, axis=0).reshape(-1, 1)
    mu_cancer = np.mean(X_cancer, axis=0).reshape(-1, 1)
    
    S_healthy = np.zeros((len(feature_names), len(feature_names)))
    S_cancer = np.zeros((len(feature_names), len(feature_names)))
    
    for sample in X_healthy:
        diff = (sample - mu_healthy.flatten()).reshape(-1, 1)
        S_healthy += diff @ diff.T
    
    for sample in X_cancer:
        diff = (sample - mu_cancer.flatten()).reshape(-1, 1)
        S_cancer += diff @ diff.T
    
    S_w = S_healthy + S_cancer
    

    S_w_inv = np.linalg.inv(S_w)


    
    # Fisher's linear discriminant direction
    w = S_w_inv @ (mu_cancer - mu_healthy)
    
    # Bias term
    b = -0.5 * (w.T @ (mu_cancer + mu_healthy))[0, 0]
    
    print(f"Healthy mean: {mu_healthy.flatten()}")
    print(f"Cancer mean: {mu_cancer.flatten()}")
    print(f"LDA weights: {w.flatten()}")
    print(f"LDA bias: {b}")


    # Print training metrics
    decision_values = (X @ w).flatten() + b
    y_pred_tr = np.zeros(len(X))
    y_pred_tr[decision_values > 0] = 1
    TP = np.sum((y_true == 1) & (y_pred_tr == 1))
    TN = np.sum((y_true == 0) & (y_pred_tr == 0))
    FP = np.sum((y_true == 0) & (y_pred_tr == 1))
    FN = np.sum((y_true == 1) & (y_pred_tr == 0))
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, decision_values)
    roc_auc = auc(fpr, tpr)
    print(f"[Train] Sensitivity (%) = {sensitivity * 100:.2f}")
    print(f"[Train] Specificity (%) = {specificity * 100:.2f}")
    print(f"[Train] Precision (%) = {precision * 100:.2f}")
    print(f"[Train] F1 Score (%) = {f1 * 100:.2f}")
    print(f"[Train] Accuracy (%) = {accuracy * 100:.2f}")
    print(f"[Train] ROC-AUC (%) = {roc_auc * 100:.2f}")

    model = {
        "w": w,
        "b": b,
        "feature_names": feature_names,
        "mu_healthy": mu_healthy,
        "mu_cancer": mu_cancer
    }

    return (
        model,
        accuracy,
        sensitivity,
        specificity,
        precision,
        f1,
        roc_auc,
    )





def test_fisher_lda(model, data=None , ixHealthy=None, ixCancer=None):


    if ixHealthy is None or ixCancer is None:
        ixHealthy = np.where(data["Classification"] == "Healthy")
        ixCancer = np.where(data["Classification"] == "Cancer")


    feature_names = model["feature_names"]
    w = model["w"]
    b = model["b"]


    X_healthy = data[feature_names].iloc[ixHealthy[0]].values
    X_cancer = data[feature_names].iloc[ixCancer[0]].values


    X = np.concatenate([X_healthy, X_cancer], axis=0)
    y_true = np.concatenate([np.zeros(len(X_healthy)), np.ones(len(X_cancer))])  # 0=Healthy, 1=Cancer


    decision_values = (X @ w).flatten() + b
    y_pred = np.zeros(len(X))
    y_pred[decision_values > 0] = 1  # Cancer class
    
    TP = np.sum((y_true == 1) & (y_pred == 1))  # Cancer correctly classified
    TN = np.sum((y_true == 0) & (y_pred == 0))  # Healthy correctly classified
    FP = np.sum((y_true == 0) & (y_pred == 1))  # Healthy misclassified as Cancer
    FN = np.sum((y_true == 1) & (y_pred == 0))  # Cancer misclassified as Healthy
    
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    fpr, tpr, _ = roc_curve(y_true, decision_values)
    roc_auc = auc(fpr, tpr)

    
    print(f"Sensitivity (%) = {sensitivity * 100:.2f}")
    print(f"Specificity (%) = {specificity * 100:.2f}")
    print(f"Precision (%) = {precision * 100:.2f}")
    print(f"F1 Score (%) = {f1_score * 100:.2f}")
    print(f"Accuracy (%) = {accuracy * 100:.2f}")
    print(f"ROC-AUC (%) = {roc_auc * 100:.2f}")

    return accuracy, sensitivity, specificity, precision, f1_score, roc_auc