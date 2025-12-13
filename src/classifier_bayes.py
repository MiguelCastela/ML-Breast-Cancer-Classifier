import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
import numpy as np
import plotly.graph_objects as go
from sklearn import mixture
from scipy.stats import multivariate_normal




def train_bayesian_classifier(feature_names, method_name="Unknown", data=None , ixHealthy=None, ixCancer=None):

    if ixHealthy is None or ixCancer is None:
        ixHealthy = np.where(data["Classification"] == "Healthy")
        ixCancer = np.where(data["Classification"] == "Cancer")
   

    print(f"\n=== BAYESIAN CLASSIFIER ({method_name}) ===")
    print(f"Using features: {feature_names}")

    X_train = data[feature_names].values
    y_train = np.concatenate([np.zeros(len(ixHealthy[0])), np.ones(len(ixCancer[0]))])
    
    ix0 = np.where(y_train == 0)[0]
    ix1 = np.where(y_train == 1)[0]


    Pw0=ix0.shape[0]/(ix0.shape[0]+ix1.shape[0])
    Pw1=ix1.shape[0]/(ix0.shape[0]+ix1.shape[0])



    clf1 = mixture.GaussianMixture(n_components=1)
    clf2 = mixture.GaussianMixture(n_components=1)
    mod1=clf1.fit(X_train[ix0,:])
    mod2=clf2.fit(X_train[ix1,:])

    mean1 = mod1.means_.squeeze()
    mean2 = mod2.means_.squeeze()

    cov1 = mod1.covariances_[0]
    cov2 = mod2.covariances_[0]

    # Regularização para evitar matrizes singulares

    reg = 1e-6
    if cov1.ndim == 0:
        cov1 = np.array([[cov1 + reg]])
        cov2 = np.array([[cov2 + reg]])
    else:
        cov1 = cov1 + reg * np.eye(cov1.shape[0])
        cov2 = cov2 + reg * np.eye(cov2.shape[0])



    model_basic = {
        "feature_names": feature_names,
        "classes": (0, 1)  # 0=Healthy, 1=Cancer
    }

    

    print(f"Healthy mean (mu_0): {mean1}")
    print(f"Cancer mean (mu_1): {mean2}")

    model = {
        "feature_names": feature_names,
        "mean0": mean1, 
        "mean1": mean2, 
        "cov0": cov1,   
        "cov1": cov2,   
        "Pw0": Pw0,     
        "Pw1": Pw1      
    }

    logp0_tr = multivariate_normal.logpdf(X_train, mean=mean1, cov=cov1) + np.log(Pw0)
    logp1_tr = multivariate_normal.logpdf(X_train, mean=mean2, cov=cov2) + np.log(Pw1)
    y_pred_tr = np.where(logp1_tr >= logp0_tr, 1, 0)
    decision_scores_tr = logp1_tr - logp0_tr
    TP = np.sum((y_train == 1) & (y_pred_tr == 1))
    TN = np.sum((y_train == 0) & (y_pred_tr == 0))
    FP = np.sum((y_train == 0) & (y_pred_tr == 1))
    FN = np.sum((y_train == 1) & (y_pred_tr == 0))
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
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


def test_bayesian_gaussian(model, data=None , ixHealthy=None, ixCancer=None):
    
    if ixHealthy is None or ixCancer is None:
        ixHealthy = np.where(data["Classification"] == "Healthy")
        ixCancer = np.where(data["Classification"] == "Cancer")

    feature_names = model["feature_names"]
    
    mean0, mean1 = model["mean0"], model["mean1"]
    cov0, cov1 = model["cov0"], model["cov1"]
    Pw0, Pw1 = model["Pw0"], model["Pw1"]
    
    X_healthy = data[feature_names].iloc[ixHealthy[0]].values
    X_cancer = data[feature_names].iloc[ixCancer[0]].values
    
    X_test = np.concatenate([X_healthy, X_cancer], axis=0)
    y_true = np.concatenate([np.zeros(len(X_healthy)), np.ones(len(X_cancer))])  

    
    # Log-likelihood + Log-prior = Log-posterior (log $P(x|w_i) + \log P(w_i)$)
    
    # P(x|w_0) * P(w_0) -> log posterior for Healthy (class 0)
    logp0 = multivariate_normal.logpdf(X_test, mean=mean0, cov=cov0) + np.log(Pw0)
    # P(x|w_1) * P(w_1) -> log posterior for Cancer (class 1)
    logp1 = multivariate_normal.logpdf(X_test, mean=mean1, cov=cov1) + np.log(Pw1)

    # Classification: $y = 1$ (Cancer) if $\log P(w_1|x) > \log P(w_0|x)$, i.e., $\log p_1 > \log p_0$.
    y_pred = np.where(logp1 >= logp0, 1, 0) # 1=Cancer, 0=Healthy
    
    
    decision_scores = logp1 - logp0 

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


def test_bayesian_custom(model, data=None , ixHealthy=None, ixCancer=None):
    

    def pdfGauss(X,mean,cov):
        covInv=np.linalg.inv(cov)
        dim=cov.shape[0]
        val=np.array([])
        for i in range(X.shape[0]):
            diff = np.array([X[i,:] - mean]) 
            # Calculate Mahalanobis distance squared: (x-mu)T * CovInv * (x-mu)
            dist = (diff @ covInv @ diff.T).squeeze() 
            # Calculate PDF value
            multivariate_pdf = np.exp(-0.5*dist)/((2*np.pi)**(dim/2)*np.linalg.det(cov)**0.5)
            val=np.append(val, multivariate_pdf)
        return np.array([val]).T
    
    def useBayes(Xte, model):
        
        
        # P(x, w0) = P(x|w0) * P(w0)
        Pw0X = pdfGauss(Xte, model['mean0'], model['cov0']) * model['Pw0']
        # P(x, w1) = P(x|w1) * P(w1)
        Pw1X = pdfGauss(Xte, model['mean1'], model['cov1']) * model['Pw1']
        
        # Decision score: Log of the posterior ratio (log(P(w1|x)/P(w0|x))) or similar value.
        # Since we use raw joint probabilities, the score is Pw1X - Pw0X.
        # NOTE: If Pw0X or Pw1X are near zero, division/subtraction can be unstable.
        
        # Use simple difference as the decision score (positive favors class 1/Cancer)
        decision_scores = (Pw1X - Pw0X).squeeze()

        # Classification: Class 1 (Cancer) if Pw1X > Pw0X, Class 0 (Healthy) otherwise
        y_pred_01 = np.where(decision_scores > 0, 1, 0)
        
        return y_pred_01, decision_scores



    if ixHealthy is None or ixCancer is None:
        ixHealthy = np.where(data["Classification"] == "Healthy")
        ixCancer = np.where(data["Classification"] == "Cancer")

    feature_names = model["feature_names"]
    
    X_healthy = data[feature_names].iloc[ixHealthy[0]].values
    X_cancer = data[feature_names].iloc[ixCancer[0]].values
    
    X_test = np.concatenate([X_healthy, X_cancer], axis=0)
    y_true = np.concatenate([np.zeros(len(X_healthy)), np.ones(len(X_cancer))])  

    y_pred, decision_scores = useBayes(X_test, model) 


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