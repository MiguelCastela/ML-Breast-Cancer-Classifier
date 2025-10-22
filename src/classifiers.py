import numpy as np
import pandas as pd
from data_loader import dados, ixHealthy, ixCancer, fnames, Classes

def euclidean_distance(feature_names, method_name="Unknown", data=None):
    if data is None:
        data = dados
    """
    Euclidean distance classifier using selected features
    """
    print(f"\n=== EUCLIDEAN DISTANCE CLASSIFIER ({method_name}) ===")
    print(f"Using features: {feature_names}")


    # Extract data for selected features
    X_healthy = data[feature_names].iloc[ixHealthy[0]].values
    X_cancer = data[feature_names].iloc[ixCancer[0]].values
    
    # Calculate means for each class
    mu_healthy = np.mean(X_healthy, axis=0).reshape(-1, 1)
    mu_cancer = np.mean(X_cancer, axis=0).reshape(-1, 1)
    
    print(f"Healthy mean: {mu_healthy.flatten()}")
    print(f"Cancer mean: {mu_cancer.flatten()}")
    
    # Combine all data
    X = np.concatenate([X_healthy, X_cancer], axis=0)
    y_true = np.concatenate([np.zeros(len(X_healthy)), np.ones(len(X_cancer))])  # 0=Healthy, 1=Cancer
    
    # Classification using euclidean distance
    y_pred = np.zeros(len(X))
    
    for i, sample in enumerate(X):
        dist_healthy = np.linalg.norm(sample - mu_healthy.flatten())
        dist_cancer = np.linalg.norm(sample - mu_cancer.flatten())
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

def mahalanobis_distance(feature_names, method_name="Unknown", data=None):
    if data is None:
        data = dados
    """
    Mahalanobis distance classifier using selected features
    """
    print(f"\n=== MAHALANOBIS DISTANCE CLASSIFIER ({method_name}) ===")
    print(f"Using features: {feature_names}")


    
    # Extract data for selected features
    X_healthy = data[feature_names].iloc[ixHealthy[0]].values
    X_cancer = data[feature_names].iloc[ixCancer[0]].values
    
    # Calculate means for each class
    mu_healthy = np.mean(X_healthy, axis=0).reshape(-1, 1)
    mu_cancer = np.mean(X_cancer, axis=0).reshape(-1, 1)
    
    if X_healthy.shape[1] == 1:
        # 1D Mahalanobis = standardized Euclidean distance
        var_healthy = np.var(X_healthy, ddof=1)
        var_cancer = np.var(X_cancer, ddof=1)
        var_pooled = (var_healthy + var_cancer) / 2
        is_1d = True
    else:
        # Calculate covariance matrices
        C_healthy = np.cov(X_healthy.T)
        C_cancer = np.cov(X_cancer.T)
        
        # Pooled covariance matrix
        C_pooled = (C_healthy + C_cancer) / 2
        
        # Handle singular matrix case
        try:
            C_inv = np.linalg.inv(C_pooled)
        except np.linalg.LinAlgError:
            # Add small regularization if matrix is singular
            C_inv = np.linalg.inv(C_pooled + np.eye(C_pooled.shape[0]) * 1e-6)
        is_1d = False
        
    print(f"Healthy mean: {mu_healthy.flatten()}")
    print(f"Cancer mean: {mu_cancer.flatten()}")
    
    # Combine all data
    X = np.concatenate([X_healthy, X_cancer], axis=0)
    y_true = np.concatenate([np.zeros(len(X_healthy)), np.ones(len(X_cancer))])  # 0=Healthy, 1=Cancer
    
    # Classification using Mahalanobis distance
    y_pred = np.zeros(len(X))
    
    for i, sample in enumerate(X):
        if is_1d:
            sample_val = sample[0]
            dist_healthy = abs(sample_val - mu_healthy.flatten()[0]) / np.sqrt(var_pooled)
            dist_cancer = abs(sample_val - mu_cancer.flatten()[0]) / np.sqrt(var_pooled)
        else:
            diff_healthy = (sample - mu_healthy.flatten()).reshape(-1, 1)
            diff_cancer = (sample - mu_cancer.flatten()).reshape(-1, 1)
            dist_healthy = np.sqrt(diff_healthy.T @ C_inv @ diff_healthy)[0, 0]
            dist_cancer = np.sqrt(diff_cancer.T @ C_inv @ diff_cancer)[0, 0]
        
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

def fisher_lda(feature_names, method_name="Unknown", data=None):
    if data is None:
        data = dados
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
    try:
        S_w_inv = np.linalg.inv(S_w)
    except np.linalg.LinAlgError:
        # Add small regularization if matrix is singular
        S_w_inv = np.linalg.inv(S_w + np.eye(S_w.shape[0]) * 1e-6)
    
    # Fisher's linear discriminant direction
    w = S_w_inv @ (mu_cancer - mu_healthy)
    
    # Bias term
    b = -0.5 * (w.T @ (mu_cancer + mu_healthy))[0, 0]
    
    print(f"Healthy mean: {mu_healthy.flatten()}")
    print(f"Cancer mean: {mu_cancer.flatten()}")
    print(f"LDA weights: {w.flatten()}")
    print(f"LDA bias: {b}")
    
    # Classification using Fisher's LDA
    y_pred = np.zeros(len(X))
    decision_values = (X @ w).flatten() + b
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


def run_all_classifiers(top5_roc=None, top5_kruskall=None, X_pca=None, LD1=None):
    """
    Run classifiers using different feature sets:
    - top5_roc / top5_kruskall: lists of column names from original dataframe
    - X_pca: numpy array of shape (n_samples, n_pcs)
    - LD1: numpy array of shape (n_samples,)
    """
    results = {}
    y_true = dados.to_numpy()[:, -1]  # True labels

    # --- ROC-AUC selected features ---
    if top5_roc is not None:
        results['roc_euclidean'] = euclidean_distance(top5_roc, "Top 5 ROC-AUC")
        results['roc_mahalanobis'] = mahalanobis_distance(top5_roc, "Top 5 ROC-AUC")
        results['roc_fisher'] = fisher_lda(top5_roc, "Top 5 ROC-AUC")

    # --- Kruskal-Wallis selected features ---
    if top5_kruskall is not None:
        results['kw_euclidean'] = euclidean_distance(top5_kruskall, "Top 5 Kruskal-Wallis")
        results['kw_mahalanobis'] = mahalanobis_distance(top5_kruskall, "Top 5 Kruskal-Wallis")
        results['kw_fisher'] = fisher_lda(top5_kruskall, "Top 5 Kruskal-Wallis")

    if X_pca is not None:
        df_pca = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])
        pca_features = list(df_pca.columns)
        
        results['pca_euclidean'] = euclidean_distance(pca_features, "Top PCA", data=df_pca)
        results['pca_mahalanobis'] = mahalanobis_distance(pca_features, "Top PCA", data=df_pca)
        results['pca_fisher'] = fisher_lda(pca_features, "Top PCA", data=df_pca)

# LDA first component (1D)
    if LD1 is not None:
        df_lda = pd.DataFrame(LD1, columns=['LD1'])
        results['lda_euclidean'] = euclidean_distance(['LD1'], "LDA1", data=df_lda)
        results['lda_fisher'] = fisher_lda(['LD1'], "LDA1", data=df_lda)
        results['lda_mahalanobis'] = mahalanobis_distance(['LD1'], "LDA1", data=df_lda)

    return results
