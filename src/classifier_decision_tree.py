import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree



def train_decision_tree(feature_names, method_name="Unknown", data=None, ixHealthy=None, ixCancer=None, max_depth=5):
    #meter estes parametros para a grid search:criterion='gini', max_depth=None,min_samples_split=2,min_samples_leaf=1,max_features=None,max_leaf_nodes=None
    """
    Trains a Decision Tree classifier.
    """
    """
    def DBPlot(m, X, y, nGrid=100):
        ## find the extent of the features
        x1_min, x1_max = X[:,0].min()-1, X[:,0].max()+1
        x2_min, x2_max = X[:,1].min()-1, X[:,1].max()+1
        ## make a dense grid (nGrid x nGrid) in this extent
        xx1, xx2=np.meshgrid(np.linspace(x1_min, x1_max, nGrid),
                            np.linspace(x2_min, x2_max, nGrid))
        XX=np.column_stack((xx1.ravel(), xx2.ravel()))
        ## predict on the grid
        hatyy=m.predict(XX).reshape(xx1.shape)
        plt.figure(figsize=(8,8))
        ## show the predictions as an semi-transparent image
        _=plt.imshow(hatyy, extent=(x1_min, x1_max, x2_min, x2_max),
                    aspect="auto",
                    interpolation='none', origin='lower',
                    alpha=0.3)
        ## add the actual data points on the image
        plt.scatter(X[:,0], X[:,1], c=y, s=30, edgecolors='k')
        plt.xlim(x1_min, x1_max)
        plt.ylim(x2_min, x2_max)
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()
    
    
    """
    if ixHealthy is None or ixCancer is None:
        ixHealthy = np.where(data["Classification"] == "Healthy")
        ixCancer = np.where(data["Classification"] == "Cancer")

    print(f"\n=== DECISION TREE CLASSIFIER (Max Depth={max_depth}, {method_name}) ===")
    print(f"Using features: {feature_names}")

    # Extract training data
    X_train = data[feature_names].values
    y_train = np.concatenate([np.zeros(len(ixHealthy[0])), np.ones(len(ixCancer[0]))]) # 0=Healthy, 1=Cancer

    # Fit Decision Tree model
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)

    model = {
        "feature_names": feature_names,
        "clf": clf,
        "max_depth": max_depth
    }

    return model

def test_decision_tree(model, data=None , ixHealthy=None, ixCancer=None):
    """
    Classifies test data using the trained Decision Tree model and computes metrics.
    """
    if ixHealthy is None or ixCancer is None:
        ixHealthy = np.where(data["Classification"] == "Healthy")
        ixCancer = np.where(data["Classification"] == "Cancer")

    feature_names = model["feature_names"]
    clf = model["clf"]

    # --- Test Data Preparation ---
    X_test = data[feature_names].values
    y_true = np.concatenate([np.zeros(len(ixHealthy[0])), np.ones(len(ixCancer[0]))])  # 0=Healthy, 1=Cancer

    # --- Prediction and Decision Scores ---
    y_pred = clf.predict(X_test)

    # Decision scores (probability of class 1 / Cancer) for ROC-AUC
    # Decision trees use predict_proba by default
    decision_scores = clf.predict_proba(X_test)[:, 1] 

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