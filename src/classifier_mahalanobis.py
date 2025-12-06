import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
import numpy as np


def train_mahalanobis_distance(feature_names, method_name="Unknown", data=None, ixHealthy=None, ixCancer=None):
    """
    Trains Mahalanobis classifier by computing class means and pooled covariance.
    """

    print(f"\n=== MAHALANOBIS DISTANCE CLASSIFIER ({method_name}) ===")
    print(f"Using features: {feature_names}")


    X_healthy = data.iloc[ixHealthy[0], :][feature_names].to_numpy()
    X_cancer  = data.iloc[ixCancer[0], :][feature_names].to_numpy()


    mu_healthy = np.mean(X_healthy, axis=0)
    mu_cancer  = np.mean(X_cancer, axis=0)

    # Pooled covariance


    if X_healthy.ndim == 1 or X_healthy.shape[1] == 1:
        # variance is scalar
        var_healthy = np.var(X_healthy, ddof=1)
        pooled_var = (np.var(X_healthy, ddof=1) + np.var(X_cancer, ddof=1)) / 2
        inv_cov = 1.0 / pooled_var
    else:
        cov_healthy = np.cov(X_healthy, rowvar=False)
        cov_cancer  = np.cov(X_cancer, rowvar=False)
        cov_pooled  = (cov_healthy + cov_cancer) / 2
        inv_cov = np.linalg.inv(cov_pooled)

    model = {
        "feature_names": feature_names,   
        "mu_healthy": mu_healthy,
        "mu_cancer": mu_cancer,
        "inv_cov": inv_cov
    }
    return model


def test_mahalanobis_distance(model, data=None , ixHealthy=None, ixCancer=None):
    """
    Classifies test data using Mahalanobis discriminant and computes metrics.
    """


    if ixHealthy is None or ixCancer is None:
        ixHealthy = np.where(data["Classification"] == "Healthy")
        ixCancer = np.where(data["Classification"] == "Cancer")

    feature_names = model["feature_names"]
    X_test = data[feature_names].to_numpy()


    y_true = np.ones(len(X_test))
    y_true[ixHealthy] = 0   # healthy
    y_true[ixCancer]  = 1   # cancer

    muH = model["mu_healthy"]
    muC = model["mu_cancer"]
    Ci  = model["inv_cov"]

    if X_test.shape[1] == 1:
        # ensure muH, muC, X_test are 1D
        muH = muH.flatten() if muH.ndim > 0 else muH
        muC = muC.flatten() if muC.ndim > 0 else muC
        X_test = X_test.flatten() if X_test.ndim > 1 else X_test
        inv_cov = Ci  # scalar
        dx = ((X_test - 0.5*(muH + muC)) * (muC - muH)) * inv_cov

    else:

    # Decision function: (muC - muH)^T * Ci * (x - 0.5*(muC + muH))
        dx = ((muC - muH).T @ Ci @ (X_test.T - 0.5*(muC + muH)[:, np.newaxis])).flatten()
    y_pred = np.zeros_like(y_true)
    y_pred[dx > 0] = 1  # classify as cancer

    # Metrics
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    sensitivity = TP / (TP + FN) if TP + FN > 0 else 0
    specificity = TN / (TN + FP) if TN + FP > 0 else 0
    precision   = TP / (TP + FP) if TP + FP > 0 else 0
    f1          = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    accuracy    = (TP + TN) / (TP + TN + FP + FN)

    fpr, tpr, _ = roc_curve(y_true, dx)
    roc_auc = auc(fpr, tpr)

    print(f"Sensitivity (%) = {sensitivity * 100:.2f}")
    print(f"Specificity (%) = {specificity * 100:.2f}")
    print(f"Precision (%) = {precision * 100:.2f}")
    print(f"F1 Score (%) = {f1 * 100:.2f}")
    print(f"Accuracy (%) = {accuracy * 100:.2f}")
    print(f"ROC-AUC (%) = {roc_auc * 100:.2f}")


    return accuracy, sensitivity, specificity, precision, f1, roc_auc