from xml.parsers.expat import model
import numpy as np
import pandas as pd
from data_loader import dados, ixHealthy, ixCancer, fnames, Classes
from scipy.spatial.distance import mahalanobis
from scipy.spatial.distance import euclidean

from scipy.spatial.distance import mahalanobis
import numpy as np



def train_euclidean_distance(feature_names, method_name="Unknown", data=None, ixHealthy=None, ixCancer=None):
    if data is None:
        data = dados
    if ixHealthy is None or ixCancer is None:
        ixHealthy = np.where(data["Classification"] == "Healthy")
        ixCancer = np.where(data["Classification"] == "Cancer")
    """
    Euclidean distance classifier using selected features
    """
    print(f"\n=== EUCLIDEAN DISTANCE CLASSIFIER ({method_name}) ===")
    print(f"Using features: {feature_names}")

    X_healthy = data[feature_names].iloc[ixHealthy[0]].values
    X_cancer = data[feature_names].iloc[ixCancer[0]].values
    # Calculate means for each class

    mu_healthy = np.mean(X_healthy, axis=0)
    mu_cancer = np.mean(X_cancer, axis=0)
    print(f"Healthy mean: {mu_healthy}")
    print(f"Cancer mean: {mu_cancer}")
    model = {
        "feature_names": feature_names,
        "mu_healthy": mu_healthy,
        "mu_cancer": mu_cancer
    }

    return model

def test_euclidean_distance(model, data=None , ixHealthy=None, ixCancer=None):
    if data is None:
        data = dados

    if ixHealthy is None or ixCancer is None:
        ixHealthy = np.where(data["Classification"] == "Healthy")
        ixCancer = np.where(data["Classification"] == "Cancer")

    feature_names = model["feature_names"]
    mu_healthy = model["mu_healthy"]
    mu_cancer = model["mu_cancer"]
    X_healthy = data[feature_names].iloc[ixHealthy[0]].values
    X_cancer = data[feature_names].iloc[ixCancer[0]].values
    # Combine all data
    X = np.concatenate([X_healthy, X_cancer], axis=0)
    y_true = np.concatenate([np.zeros(len(X_healthy)), np.ones(len(X_cancer))])  # 0=Healthy, 1=Cancer
    # Classification using euclidean distance
    y_pred = np.zeros(len(X))
    for i, sample in enumerate(X):
        dist_healthy = np.linalg.norm(sample - mu_healthy)
        dist_cancer = np.linalg.norm(sample - mu_cancer)
        y_pred[i] = 0 if dist_healthy < dist_cancer else 1
    # Calculate metrics
    TP = np.sum((y_true == 1) & (y_pred == 1))  # Cancer correctly classified
    TN = np.sum((y_true == 0) & (y_pred == 0))  # Healthy correctly classified
    FP = np.sum((y_true == 0) & (y_pred == 1))  # Healthy misclassified as Cancer
    FN = np.sum((y_true == 1) & (y_pred == 0))  # Cancer misclassified as Healthy
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print(f"Sensitivity (%) = {sensitivity * 100:.2f}")
    print(f"Specificity (%) = {specificity * 100:.2f}")
    print(f"Precision (%) = {precision * 100:.2f}")
    print(f"F1 Score (%) = {f1_score * 100:.2f}")
    print(f"Accuracy (%) = {accuracy * 100:.2f}")
    return accuracy, sensitivity, specificity, precision, f1_score


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
    if data is None:
        data = dados

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

    print(f"Sensitivity (%) = {sensitivity * 100:.2f}")
    print(f"Specificity (%) = {specificity * 100:.2f}")
    print(f"Precision (%) = {precision * 100:.2f}")
    print(f"F1 Score (%) = {f1 * 100:.2f}")
    print(f"Accuracy (%) = {accuracy * 100:.2f}")


    return accuracy, sensitivity, specificity, precision, f1



def train_fisher_lda(feature_names, method_name="Unknown", data=None , ixHealthy=None, ixCancer=None):
    if data is None:
        data = dados

    if ixHealthy is None or ixCancer is None:
        ixHealthy = np.where(data["Classification"] == "Healthy")
        ixCancer = np.where(data["Classification"] == "Cancer")
    """
    Fisher's Linear Discriminant Analysis classifier using selected features
    """
    print(f"\n=== FISHER'S LDA CLASSIFIER ({method_name}) ===")
    print(f"Using features: {feature_names}")


    
    # Extract data for selected features
    X_healthy = data[feature_names].iloc[ixHealthy[0]].values
    X_cancer = data[feature_names].iloc[ixCancer[0]].values
    
    # Combine all data
    X = np.concatenate([X_healthy, X_cancer], axis=0)
    y_true = np.concatenate([np.zeros(len(X_healthy)), np.ones(len(X_cancer))])  # 0=Healthy, 1=Cancer
    
    # Calculate means for each class
    mu_healthy = np.mean(X_healthy, axis=0).reshape(-1, 1)
    mu_cancer = np.mean(X_cancer, axis=0).reshape(-1, 1)
    
    # Calculate within-class scatter matrices
    S_healthy = np.zeros((len(feature_names), len(feature_names)))
    S_cancer = np.zeros((len(feature_names), len(feature_names)))
    
    # Within-class scatter for healthy samples
    for sample in X_healthy:
        diff = (sample - mu_healthy.flatten()).reshape(-1, 1)
        S_healthy += diff @ diff.T
    
    # Within-class scatter for cancer samples
    for sample in X_cancer:
        diff = (sample - mu_cancer.flatten()).reshape(-1, 1)
        S_cancer += diff @ diff.T
    
    # Total within-class scatter matrix
    S_w = S_healthy + S_cancer
    
    # Handle singular matrix case
    if len(feature_names) == 1:
        # within-class scatter as sum of variances
        S_w_scalar = np.var(X_healthy, ddof=1) * (len(X_healthy)-1) + np.var(X_cancer, ddof=1) * (len(X_cancer)-1)
        S_w_inv = np.array([[1.0 / S_w_scalar]])
    else:
        S_w_inv = np.linalg.inv(S_w)


    
    # Fisher's linear discriminant direction
    w = S_w_inv @ (mu_cancer - mu_healthy)
    
    # Bias term
    b = -0.5 * (w.T @ (mu_cancer + mu_healthy))[0, 0]
    
    print(f"Healthy mean: {mu_healthy.flatten()}")
    print(f"Cancer mean: {mu_cancer.flatten()}")
    print(f"LDA weights: {w.flatten()}")
    print(f"LDA bias: {b}")


    return{
        "w": w,
        "b": b,
        "feature_names": feature_names,
        "mu_healthy": mu_healthy,
        "mu_cancer": mu_cancer
    }



def test_fisher_lda(model, data=None , ixHealthy=None, ixCancer=None):

    if data is None:
        data = dados

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


    # Classification using Fisher's LDA
    X = X.reshape(len(X), -1)  # (n_samples, n_features)

# Ensure w is 2D column
    w = model["w"].reshape(-1, 1)  # (n_features, 1)


    decision_values = (X @ w).flatten() + b
    y_pred = np.zeros(len(X))
    y_pred[decision_values > 0] = 1  # Cancer class
    
    # Calculate metrics
    TP = np.sum((y_true == 1) & (y_pred == 1))  # Cancer correctly classified
    TN = np.sum((y_true == 0) & (y_pred == 0))  # Healthy correctly classified
    FP = np.sum((y_true == 0) & (y_pred == 1))  # Healthy misclassified as Cancer
    FN = np.sum((y_true == 1) & (y_pred == 0))  # Cancer misclassified as Healthy
    
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    print(f"Sensitivity (%) = {sensitivity * 100:.2f}")
    print(f"Specificity (%) = {specificity * 100:.2f}")
    print(f"Precision (%) = {precision * 100:.2f}")
    print(f"F1 Score (%) = {f1_score * 100:.2f}")
    print(f"Accuracy (%) = {accuracy * 100:.2f}")
    
    return accuracy, sensitivity, specificity, precision, f1_score


def run_all_classifiers(train_data, test_data, ixHealthy_train, ixCancer_train, ixHealthy_test, ixCancer_test, all_features, top5_roc=None, top5_kruskall=None, X_pca_train=None, X_pca_test=None, LD1_train=None, LD1_test=None):
   

    """
    Run classifiers using different feature sets and train/test split:
    - train_data / test_data: pandas DataFrames with the same columns
    - top5_roc / top5_kruskall: lists of column names from original dataframe
    - X_pca_train / X_pca_test: numpy arrays for PCA features
    - LD1_train / LD1_test: numpy arrays for LDA1 (1D) features
    """
    results = {}




    # Euclidean
    model = train_euclidean_distance(all_features, method_name="All Features", 
                                    data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
    results['all_euclidean'] = test_euclidean_distance(model, data=test_data, 
                                                    ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

    # Mahalanobis
    model = train_mahalanobis_distance(all_features, method_name="All Features", 
                                    data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
    results['all_mahalanobis'] = test_mahalanobis_distance(model, data=test_data, 
                                                        ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

    # Fisher LDA
    model = train_fisher_lda(all_features, method_name="All Features", 
                            data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
    results['all_fisher'] = test_fisher_lda(model, data=test_data, 
                                            ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)


    # --- ROC-AUC selected features ---
    
    if top5_roc is not None:
        model = train_euclidean_distance(top5_roc, method_name="ROC-AUC", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['roc_euclidean'] = test_euclidean_distance(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_mahalanobis_distance(top5_roc, method_name="ROC-AUC", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['roc_mahalanobis'] = test_mahalanobis_distance(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_fisher_lda(top5_roc, method_name="ROC-AUC", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['roc_fisher'] = test_fisher_lda(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

    # --- Kruskal-Wallis selected features ---
    if top5_kruskall is not None:
        model = train_euclidean_distance(top5_kruskall, method_name="Kruskal-Wallis", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['kw_euclidean'] = test_euclidean_distance(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_mahalanobis_distance(top5_kruskall, method_name="Kruskal-Wallis", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['kw_mahalanobis'] = test_mahalanobis_distance(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_fisher_lda(top5_kruskall, method_name="Kruskal-Wallis", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['kw_fisher'] = test_fisher_lda(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

    # --- PCA features ---
    if X_pca_train is not None and X_pca_test is not None:
        df_pca_train = pd.DataFrame(X_pca_train, columns=[f"PC{i+1}" for i in range(X_pca_train.shape[1])])
        df_pca_test = pd.DataFrame(X_pca_test, columns=[f"PC{i+1}" for i in range(X_pca_test.shape[1])])
        pca_features = list(df_pca_train.columns)

        model = train_euclidean_distance(pca_features, method_name="PCA", data=df_pca_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['pca_euclidean'] = test_euclidean_distance(model, data=df_pca_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_mahalanobis_distance(pca_features, method_name="PCA", data=df_pca_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['pca_mahalanobis'] = test_mahalanobis_distance(model, data=df_pca_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_fisher_lda(pca_features, method_name="PCA", data=df_pca_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['pca_fisher'] = test_fisher_lda(model, data=df_pca_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

    # --- LDA first component (1D) ---
    if LD1_train is not None and LD1_test is not None:


        df_lda_train = pd.DataFrame(LD1_train, columns=['LD1'])
        df_lda_test = pd.DataFrame(LD1_test, columns=['LD1'])
    

        model = train_euclidean_distance(['LD1'], method_name="LDA", data=df_lda_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['lda_euclidean'] = test_euclidean_distance(model, data=df_lda_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_mahalanobis_distance(['LD1'], method_name="LDA", data=df_lda_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['lda_mahalanobis'] = test_mahalanobis_distance(model, data=df_lda_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_fisher_lda(['LD1'], method_name="LDA", data=df_lda_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['lda_fisher'] = test_fisher_lda(model, data=df_lda_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

    return results