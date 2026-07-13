import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
import numpy as np


def train_euclidean_distance(feature_names, method_name="Unknown", data=None, ixHealthy=None, ixCancer=None):

    if ixHealthy is None or ixCancer is None:
        ixHealthy = np.where(data["Classification"] == "Healthy")
        ixCancer = np.where(data["Classification"] == "Cancer")
    
    print(f"\n=== EUCLIDEAN DISTANCE CLASSIFIER ({method_name}) ===")
    print(f"Using features: {feature_names}")

    X_healthy = data[feature_names].iloc[ixHealthy[0]].values
    X_cancer = data[feature_names].iloc[ixCancer[0]].values

    mu_healthy = np.mean(X_healthy, axis=0)
    mu_cancer = np.mean(X_cancer, axis=0)
    print(f"Healthy mean: {mu_healthy}")
    print(f"Cancer mean: {mu_cancer}")
    model = {
        "feature_names": feature_names,
        "mu_healthy": mu_healthy,
        "mu_cancer": mu_cancer
    }

    X = np.concatenate([X_healthy, X_cancer], axis=0)
    y_train = np.concatenate([np.zeros(len(X_healthy)), np.ones(len(X_cancer))])
    dist_healthy = np.linalg.norm(X - mu_healthy, axis=1)
    dist_cancer = np.linalg.norm(X - mu_cancer, axis=1)
    y_pred_tr = (dist_cancer < dist_healthy).astype(int)
    TP = np.sum((y_train == 1) & (y_pred_tr == 1))
    TN = np.sum((y_train == 0) & (y_pred_tr == 0))
    FP = np.sum((y_train == 0) & (y_pred_tr == 1))
    FN = np.sum((y_train == 1) & (y_pred_tr == 0))
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    decision_scores = dist_healthy - dist_cancer
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_train, decision_scores)
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

def test_euclidean_distance(model, data=None , ixHealthy=None, ixCancer=None):

    if ixHealthy is None or ixCancer is None:
        ixHealthy = np.where(data["Classification"] == "Healthy")
        ixCancer = np.where(data["Classification"] == "Cancer")

    feature_names = model["feature_names"]
    mu_healthy = model["mu_healthy"]
    mu_cancer = model["mu_cancer"]
    X_healthy = data[feature_names].iloc[ixHealthy[0]].values
    X_cancer = data[feature_names].iloc[ixCancer[0]].values
    X = np.concatenate([X_healthy, X_cancer], axis=0)
    y_true = np.concatenate([np.zeros(len(X_healthy)), np.ones(len(X_cancer))])  

    dist_healthy = np.zeros(len(X))
    dist_cancer = np.zeros(len(X))
    y_pred = np.zeros(len(X))
    for i, sample in enumerate(X):
        dist_healthy[i] = np.linalg.norm(sample - mu_healthy)
        dist_cancer[i] = np.linalg.norm(sample - mu_cancer)
        y_pred[i] = 0 if dist_healthy[i] < dist_cancer[i] else 1

    TP = np.sum((y_true == 1) & (y_pred == 1))  
    TN = np.sum((y_true == 0) & (y_pred == 0))  
    FP = np.sum((y_true == 0) & (y_pred == 1))  
    FN = np.sum((y_true == 1) & (y_pred == 0))  
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    decision_scores = dist_healthy - dist_cancer

    fpr, tpr, _ = roc_curve(y_true, decision_scores)
    roc_auc = auc(fpr, tpr)


    print(f"Sensitivity (%) = {sensitivity * 100:.2f}")
    print(f"Specificity (%) = {specificity * 100:.2f}")
    print(f"Precision (%) = {precision * 100:.2f}")
    print(f"F1 Score (%) = {f1_score * 100:.2f}")
    print(f"Accuracy (%) = {accuracy * 100:.2f}")
    print(f"ROC-AUC (%) = {roc_auc * 100:.2f}")
    return accuracy, sensitivity, specificity, precision, f1_score, roc_auc
