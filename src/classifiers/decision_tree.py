import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree



def train_decision_tree(feature_names, method_name="Unknown", data=None, ixHealthy=None, ixCancer=None, max_depth=5, criterion='gini'):

    
    if ixHealthy is None or ixCancer is None:
        ixHealthy = np.where(data["Classification"] == "Healthy")
        ixCancer = np.where(data["Classification"] == "Cancer")

    print(f"\n=== DECISION TREE CLASSIFIER (Max Depth={max_depth}, {method_name}) ===")
    print(f"Using features: {feature_names}")

    X_train = data[feature_names].values
    y_train = np.concatenate([np.zeros(len(ixHealthy[0])), np.ones(len(ixCancer[0]))]) # 0=Healthy, 1=Cancer

    clf = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)
    clf.fit(X_train, y_train)

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

    model = {
        "feature_names": feature_names,
        "clf": clf,
        "max_depth": max_depth
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

def test_decision_tree(model, data=None , ixHealthy=None, ixCancer=None):
    
    if ixHealthy is None or ixCancer is None:
        ixHealthy = np.where(data["Classification"] == "Healthy")
        ixCancer = np.where(data["Classification"] == "Cancer")

    feature_names = model["feature_names"]
    clf = model["clf"]

    X_test = data[feature_names].values
    y_true = np.concatenate([np.zeros(len(ixHealthy[0])), np.ones(len(ixCancer[0]))])  

    y_pred = clf.predict(X_test)

    # Decision scores (probability of class 1 / Cancer) for ROC-AUC
    # Decision trees use predict_proba by default
    decision_scores = clf.predict_proba(X_test)[:, 1] 

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