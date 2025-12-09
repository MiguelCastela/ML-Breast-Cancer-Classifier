
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import DecisionBoundaryDisplay

def train_knn(feature_names, method_name="Unknown", data=None, ixHealthy=None, ixCancer=None, n_neighbors=5):
    """
    Trains a K-Nearest Neighbors classifier (non-parametric model).
    It fits the model and stores the feature names and the trained classifier object.
    (Note: Uses a fixed k=5 for consistency with the train/test framework.)
    """
    if ixHealthy is None or ixCancer is None:
        ixHealthy = np.where(data["Classification"] == "Healthy")
        ixCancer = np.where(data["Classification"] == "Cancer")

    print(f"\n=== KNN CLASSIFIER (K={n_neighbors}, {method_name}) ===")
    print(f"Using features: {feature_names}")

    # Extract training data
    X_train = data[feature_names].values
    y_train = np.concatenate([np.zeros(len(ixHealthy[0])), np.ones(len(ixCancer[0]))]) # 0=Healthy, 1=Cancer

    # Fit KNN model
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)

    model = {
        "feature_names": feature_names,
        "clf": clf,
        "n_neighbors": n_neighbors
    }

    return model

def test_knn(model, data=None , ixHealthy=None, ixCancer=None):
    """
    Classifies test data using the trained KNN model and computes metrics.
    """
    if ixHealthy is None or ixCancer is None:
        ixHealthy = np.where(data["Classification"] == "Healthy")
        ixCancer = np.where(data["Classification"] == "Cancer")

    feature_names = model["feature_names"]
    clf = model["clf"]
    n_neighbors = model["n_neighbors"]

    # --- Test Data Preparation ---
    X_test = data[feature_names].values
    y_true = np.concatenate([np.zeros(len(ixHealthy[0])), np.ones(len(ixCancer[0]))])  # 0=Healthy, 1=Cancer

    # --- Prediction and Decision Scores ---
    y_pred = clf.predict(X_test)

    # Decision scores (probability of class 1 / Cancer) for ROC-AUC
    # We use predict_proba for ROC-AUC as the distance isn't easily interpretable as a score.
    decision_scores = clf.predict_proba(X_test)[:, 1] 

    # --- Metric Calculation (Consistent with other test functions) ---
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



