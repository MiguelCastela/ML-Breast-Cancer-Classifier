import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
import numpy as np
from sklearn.tree import DecisionTreeClassifier


def train_adaboost(feature_names, method_name="Unknown", data=None, ixHealthy=None, ixCancer=None, n_estimators=5, learning_rate=1.0):
    
    if ixHealthy is None or ixCancer is None:
        ixHealthy = np.where(data["Classification"] == "Healthy")
        ixCancer = np.where(data["Classification"] == "Cancer")

    print(f"\n=== ADABOOST CLASSIFIER (n_estimators={n_estimators}, {method_name}) ===")
    print(f"Using features: {feature_names}")

    X_train = data[feature_names].values
    y_train = np.concatenate([np.zeros(len(ixHealthy[0])), np.ones(len(ixCancer[0]))]) 

    base_estimator = DecisionTreeClassifier(max_depth=1) 
    

    clf = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=42,
        algorithm='SAMME'
    )
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
        "n_estimators": n_estimators
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

from sklearn.metrics import roc_curve, auc

def test_adaboost(model, data=None , ixHealthy=None, ixCancer=None):
    
    if ixHealthy is None or ixCancer is None:
        ixHealthy = np.where(data["Classification"] == "Healthy")
        ixCancer = np.where(data["Classification"] == "Cancer")

    feature_names = model["feature_names"]
    clf = model["clf"]

    X_test = data[feature_names].values
    y_true = np.concatenate([np.zeros(len(ixHealthy[0])), np.ones(len(ixCancer[0]))])  

    y_pred = clf.predict(X_test)

    
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

# map 0/1 to -1/+1
def map_labels_to_pm1(y):
    return np.where(y == 1, 1, -1)

def train_adaboost_custom(feature_names, method_name="Unknown", data=None, ixHealthy=None, ixCancer=None, n_mod=5, learning_rate=1.0):
    
    if ixHealthy is None or ixCancer is None:
        ixHealthy = np.where(data["Classification"] == "Healthy")
        ixCancer = np.where(data["Classification"] == "Cancer")

    print(f"\n=== CUSTOM ADABOOST CLASSIFIER (n_mod={n_mod}, {method_name}) ===")
    print(f"Using features: {feature_names}")

    X_train = data[feature_names].values
    y_train_01 = np.concatenate([np.zeros(len(ixHealthy[0])), np.ones(len(ixCancer[0]))])
    y_train_pm1 = map_labels_to_pm1(y_train_01) 

    
    def trainAdaboost(Xtr,ytr,nMod, learning_rate):
        models= np.arange(nMod, dtype=DecisionTreeClassifier)
        N=Xtr.shape[0]
        w=np.ones((N))*(1/N)
        alphas=np.zeros((nMod))
        
        for i in range(nMod):
            mdl=DecisionTreeClassifier(criterion='gini',max_depth=1) 
            mdl.fit(Xtr,ytr,sample_weight=w)
            pred=mdl.predict(Xtr)
            models[i]=mdl
            
            
            err=sum(w*np.heaviside(-ytr*pred,1)) 
            
            # Handle near-zero error to prevent log(0)
            if err < 1e-10: 
                err = 1e-10
            elif err >= 1.0 - 1e-10:
                err = 1.0 - 1e-10

            alpha_classic=0.5*np.log((1-err)/err)
            alpha=learning_rate*alpha_classic
            models[i].alpha=alpha 
            
            # Weight update: w_j <- w_j * exp(-alpha * y_j * h(x_j))
            v=np.zeros((N))
            for j in range(N):
                v[j]=w[j]*np.exp(-alpha_classic*ytr[j]*pred[j])
            
            Sm=np.sum(v);
            w=v/Sm; 
            
        return models
    
    trained_models = trainAdaboost(X_train, y_train_pm1, n_mod, learning_rate)
    
   

    model = {
        "feature_names": feature_names,
        "models": trained_models, 
        "n_mod": n_mod
    }


    nMod = trained_models.shape[0]
    nsamp = X_train.shape[0]
    predTot = np.zeros((nMod, nsamp))
    for m in range(nMod):
        label = trained_models[m].predict(X_train)  
        predTot[m, :] = trained_models[m].alpha * label
    decision_scores_tr = np.sum(predTot, axis=0)
    y_pred_pm1 = np.sign(decision_scores_tr)
    y_pred_tr = map_labels_to_01(y_pred_pm1)
    y_train = y_train_01 

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


# map -1/+1 back to 0/1
def map_labels_to_01(y):
    
    return np.where(y == 1, 1, 0)

def test_adaboost_custom(model, data=None , ixHealthy=None, ixCancer=None):
    
    if ixHealthy is None or ixCancer is None:
        ixHealthy = np.where(data["Classification"] == "Healthy")
        ixCancer = np.where(data["Classification"] == "Cancer")

    feature_names = model["feature_names"]
    models = model["models"]
    n_mod = model["n_mod"]

    X_test = data[feature_names].values
    y_true = np.concatenate([np.zeros(len(ixHealthy[0])), np.ones(len(ixCancer[0]))])  

    
    def testAdaboost(Xte,models): # Removed yte as it's not used in prediction logic
        
        nMod=models.shape[0]
        nsamp=Xte.shape[0]
        
        predTot=np.zeros((nMod,nsamp))
        
        
        for m in range(nMod):
            label=models[m].predict(Xte) 
            
            predTot[m,:]=models[m].alpha*label 
            
        # Decision score is the sum of weighted predictions: F(x) = sum(alpha_m * h_m(x))
        decision_scores = np.sum(predTot,0)
        
        y_pred_pm1 = np.sign(decision_scores) 
        
        return y_pred_pm1, decision_scores 
    
    y_pred_pm1, decision_scores = testAdaboost(X_test, models)
    
    y_pred = map_labels_to_01(y_pred_pm1)



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
