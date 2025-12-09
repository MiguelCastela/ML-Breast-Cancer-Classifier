import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
import numpy as np
from sklearn.tree import DecisionTreeClassifier


def train_adaboost(feature_names, method_name="Unknown", data=None, ixHealthy=None, ixCancer=None, n_estimators=5, learning_rate=1.0):
    """
    Trains an AdaBoost (Adaptive Boosting) classifier using Decision Trees as weak learners.
    """
    if ixHealthy is None or ixCancer is None:
        ixHealthy = np.where(data["Classification"] == "Healthy")
        ixCancer = np.where(data["Classification"] == "Cancer")

    print(f"\n=== ADABOOST CLASSIFIER (n_estimators={n_estimators}, {method_name}) ===")
    print(f"Using features: {feature_names}")

    # Extract training data
    X_train = data[feature_names].values
    y_train = np.concatenate([np.zeros(len(ixHealthy[0])), np.ones(len(ixCancer[0]))]) # 0=Healthy, 1=Cancer

    # Define the base estimator (weak learner)
    base_estimator = DecisionTreeClassifier(max_depth=1) 
    
    # Fit the AdaBoost model
    # Use a supported algorithm variant. Newer scikit-learn versions deprecate/limit 'SAMME.R'.
    # 'SAMME' is classification-oriented and supported.
    clf = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=42,
        algorithm='SAMME'
    )
    clf.fit(X_train, y_train)

    model = {
        "feature_names": feature_names,
        "clf": clf,
        "n_estimators": n_estimators
    }

    return model

from sklearn.metrics import roc_curve, auc

def test_adaboost(model, data=None , ixHealthy=None, ixCancer=None):
    """
    Classifies test data using the trained AdaBoost model and computes metrics.
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
    # AdaBoost uses predict_proba
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

# Helper function to map 0/1 to -1/+1
def map_labels_to_pm1(y):
    # Converts 0 (Healthy) -> -1 and 1 (Cancer) -> 1
    return np.where(y == 1, 1, -1)

def train_adaboost_custom(feature_names, method_name="Unknown", data=None, ixHealthy=None, ixCancer=None, n_mod=5):
    """
    Trains a custom AdaBoost classifier using the provided logic. 
    Requires y labels to be {-1, +1}.
    """
    if ixHealthy is None or ixCancer is None:
        ixHealthy = np.where(data["Classification"] == "Healthy")
        ixCancer = np.where(data["Classification"] == "Cancer")

    print(f"\n=== CUSTOM ADABOOST CLASSIFIER (n_mod={n_mod}, {method_name}) ===")
    print(f"Using features: {feature_names}")

    # --- Data Extraction and Label Conversion ---
    X_train = data[feature_names].values
    y_train_01 = np.concatenate([np.zeros(len(ixHealthy[0])), np.ones(len(ixCancer[0]))])
    y_train_pm1 = map_labels_to_pm1(y_train_01) # Convert to -1/+1

    # -------------------------------------------------------------------------
    # START: Adapted Core Training Logic from your code
    
    def trainAdaboost(Xtr,ytr,nMod):
        models= np.arange(nMod, dtype=DecisionTreeClassifier)
        N=Xtr.shape[0]
        w=np.ones((N))*(1/N)
        alphas=np.zeros((nMod))
        
        for i in range(nMod):
            # Base estimator must be a decision stump (max_depth=1) for classic AdaBoost
            mdl=DecisionTreeClassifier(criterion='gini',max_depth=1) 
            mdl.fit(Xtr,ytr,sample_weight=w)
            pred=mdl.predict(Xtr)
            models[i]=mdl
            
            # Error calculation: err = sum(w * I(y_i != h(x_i)))
            # I(y_i != h(x_i)) is equivalent to heaviside(-y_i * h(x_i), 1) when y_i, h(x_i) in {-1, 1}
            err=sum(w*np.heaviside(-ytr*pred,1)) 
            
            # Handle near-zero error to prevent log(0)
            if err < 1e-10: 
                err = 1e-10
            elif err >= 1.0 - 1e-10:
                err = 1.0 - 1e-10

            alpha=0.5*np.log((1-err)/err)
            models[i].alpha=alpha # Store alpha with the model
            
            # Weight update: w_j <- w_j * exp(-alpha * y_j * h(x_j))
            v=np.zeros((N))
            for j in range(N):
                v[j]=w[j]*np.exp(-alpha*ytr[j]*pred[j])
            
            Sm=np.sum(v);
            w=v/Sm; # Normalize weights
            
        return models
    
    # Run the custom training function
    trained_models = trainAdaboost(X_train, y_train_pm1, n_mod)
    
    # END: Adapted Core Training Logic
    # -------------------------------------------------------------------------

    model = {
        "feature_names": feature_names,
        "models": trained_models, # Array of decision stumps with stored alpha
        "n_mod": n_mod
    }

    return model


# Helper function to map -1/+1 back to 0/1
def map_labels_to_01(y):
    # Converts -1 -> 0 and 1 -> 1
    return np.where(y == 1, 1, 0)

def test_adaboost_custom(model, data=None , ixHealthy=None, ixCancer=None):
    """
    Classifies test data using the trained custom AdaBoost model and computes metrics.
    """
    if ixHealthy is None or ixCancer is None:
        ixHealthy = np.where(data["Classification"] == "Healthy")
        ixCancer = np.where(data["Classification"] == "Cancer")

    feature_names = model["feature_names"]
    models = model["models"]
    n_mod = model["n_mod"]

    # --- Test Data Preparation ---
    X_test = data[feature_names].values
    y_true = np.concatenate([np.zeros(len(ixHealthy[0])), np.ones(len(ixCancer[0]))])  # True labels in 0/1

    # -------------------------------------------------------------------------
    # START: Adapted Core Testing Logic from your code
    
    def testAdaboost(Xte,models): # Removed yte as it's not used in prediction logic
        
        nMod=models.shape[0]
        nsamp=Xte.shape[0]
        
        predTot=np.zeros((nMod,nsamp))
        
        # decisionTot is not necessary if we calculate scores and prediction in one go
        # decisionTot=np.zeros((nMod,nsamp)) 
        
        for m in range(nMod):
            # Weak classifier predicts labels in {-1, 1}
            label=models[m].predict(Xte) 
            
            # The contribution of the m-th model to the final decision score is alpha * label
            predTot[m,:]=models[m].alpha*label 
            
        # Decision score is the sum of weighted predictions: F(x) = sum(alpha_m * h_m(x))
        decision_scores = np.sum(predTot,0)
        
        # Final prediction: sign(F(x)). Result is in {-1, 1}.
        y_pred_pm1 = np.sign(decision_scores) 
        
        return y_pred_pm1, decision_scores # Return both for metrics
    
    # Run the custom testing function
    y_pred_pm1, decision_scores = testAdaboost(X_test, models)
    
    # Final prediction must be converted back to 0/1 for metric calculation
    y_pred = map_labels_to_01(y_pred_pm1)

    # END: Adapted Core Testing Logic
    # -------------------------------------------------------------------------


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

    # decision_scores (F(x)) are the raw values for ROC, higher F(x) favors class 1 (Cancer)
    fpr, tpr, _ = roc_curve(y_true, decision_scores)
    roc_auc = auc(fpr, tpr)

    print(f"Sensitivity (%) = {sensitivity * 100:.2f}")
    print(f"Specificity (%) = {specificity * 100:.2f}")
    print(f"Precision (%) = {precision * 100:.2f}")
    print(f"F1 Score (%) = {f1_score * 100:.2f}")
    print(f"Accuracy (%) = {accuracy * 100:.2f}")
    print(f"ROC-AUC (%) = {roc_auc * 100:.2f}")

    return accuracy, sensitivity, specificity, precision, f1_score, roc_auc
