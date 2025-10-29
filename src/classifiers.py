import numpy as np
import pandas as pd
from data_loader import dados, ixHealthy, ixCancer, fnames, Classes
from scipy.spatial.distance import mahalanobis



    # Extract data for selected features
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
    mu_healthy = np.mean(X_healthy, axis=0).reshape(-1, 1)
    mu_cancer = np.mean(X_cancer, axis=0).reshape(-1, 1)
    
    print(f"Healthy mean: {mu_healthy.flatten()}")
    print(f"Cancer mean: {mu_cancer.flatten()}")

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

def train_mahalanobis_distance_(feature_names, method_name="Unknown", data=None, ixHealthy=None, ixCancer=None):
    if data is None:
        data = dados

    if ixHealthy is None or ixCancer is None:
        ixHealthy = np.where(data["Classification"] == "Healthy")
        ixCancer = np.where(data["Classification"] == "Cancer")
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
        model = {   
            "is_1d": True,
            "var_pooled": var_pooled
        }
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
        model = {
            "is_1d": False,
            "C_inv": C_inv
        }
        
    print(f"Healthy mean: {mu_healthy.flatten()}")
    print(f"Cancer mean: {mu_cancer.flatten()}")


    model.update({
        "feature_names": feature_names,
        "mu_healthy": mu_healthy,
        "mu_cancer": mu_cancer,
        })

    return model
    


def test_mahalanobis_distance_(model, data=None , ixHealthy=None, ixCancer=None):

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
    
    # Classification using Mahalanobis distance
    y_pred = np.zeros(len(X))
    if model["is_1d"]:   
        var_pooled = model["var_pooled"]
        for i, sample in enumerate(X):
                sample_val = sample[0]
                dist_healthy = abs(sample_val - mu_healthy.flatten()[0]) / np.sqrt(var_pooled)
                dist_cancer = abs(sample_val - mu_cancer.flatten()[0]) / np.sqrt(var_pooled)
    else:
        C_inv = model["C_inv"]
        for i, sample in enumerate(X):
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


def train_mahalanobis_distance(feature_names, method_name="Unknown", data=None, ixHealthy=None, ixCancer=None):
    if data is None:
        data = dados

    if ixHealthy is None or ixCancer is None:
        ixHealthy = np.where(data["Classification"] == "Healthy")
        ixCancer = np.where(data["Classification"] == "Cancer")

    print(f"\n=== MAHALANOBIS DISTANCE CLASSIFIER ({method_name}) ===")
    print(f"Using features: {feature_names}")

    # Extract data for selected features
    X_healthy = data[feature_names].iloc[ixHealthy[0]].values
    X_cancer = data[feature_names].iloc[ixCancer[0]].values

    # Calculate means for each class
    mu_healthy = np.mean(X_healthy, axis=0)
    mu_cancer = np.mean(X_cancer, axis=0)

    # Covariance matrices and pooled covariance
    if X_healthy.shape[1] == 1:
        var_healthy = np.var(X_healthy, ddof=1)
        var_cancer = np.var(X_cancer, ddof=1)
        var_pooled = (var_healthy + var_cancer) / 2
        VI = 1.0 / var_pooled  # Inverse variance for 1D
        is_1d = True
    else:
        C_healthy = np.cov(X_healthy.T)
        C_cancer = np.cov(X_cancer.T)
        C_pooled = (C_healthy + C_cancer) / 2

        # Regularize if necessary
        try:
            VI = np.linalg.inv(C_pooled)
        except np.linalg.LinAlgError:
            VI = np.linalg.inv(C_pooled + np.eye(C_pooled.shape[0]) * 1e-6)
        is_1d = False

    print(f"Healthy mean: {mu_healthy}")
    print(f"Cancer mean: {mu_cancer}")

    model = {
        "feature_names": feature_names,
        "mu_healthy": mu_healthy,
        "mu_cancer": mu_cancer,
        "VI": VI,
        "is_1d": is_1d
    }

    return model


def test_mahalanobis_distance(model, data=None, ixHealthy=None, ixCancer=None):
    if data is None:
        data = dados

    if ixHealthy is None or ixCancer is None:
        ixHealthy = np.where(data["Classification"] == "Healthy")
        ixCancer = np.where(data["Classification"] == "Cancer")

    feature_names = model["feature_names"]
    mu_healthy = model["mu_healthy"]
    mu_cancer = model["mu_cancer"]
    VI = model["VI"]

    X_healthy = data[feature_names].iloc[ixHealthy[0]].values
    X_cancer = data[feature_names].iloc[ixCancer[0]].values

    X = np.concatenate([X_healthy, X_cancer], axis=0)
    y_true = np.concatenate([
        np.zeros(len(X_healthy)),  # 0 = Healthy
        np.ones(len(X_cancer))     # 1 = Cancer
    ])

    y_pred = np.zeros(len(X))

    for i, sample in enumerate(X):
        if model["is_1d"]:
            dist_healthy = np.abs(sample - mu_healthy) / np.sqrt(1.0 / VI)
            dist_cancer = np.abs(sample - mu_cancer) / np.sqrt(1.0 / VI)
        else:
            dist_healthy = mahalanobis(sample, mu_healthy, VI)
            dist_cancer = mahalanobis(sample, mu_cancer, VI)

        y_pred[i] = 0 if dist_healthy < dist_cancer else 1

    # Metrics
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

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


def run_all_classifiers(train_data, test_data, ixHealthy_train, ixCancer_train, ixHealthy_test, ixCancer_test, top5_roc=None, top5_kruskall=None, X_pca_train=None, X_pca_test=None, LD1_train=None, LD1_test=None):
   

    """
    Run classifiers using different feature sets and train/test split:
    - train_data / test_data: pandas DataFrames with the same columns
    - top5_roc / top5_kruskall: lists of column names from original dataframe
    - X_pca_train / X_pca_test: numpy arrays for PCA features
    - LD1_train / LD1_test: numpy arrays for LDA1 (1D) features
    """
    results = {}


    # Select all features (exclude the label column)
    all_features = [col for col in train_data.columns if col != "Classification"]

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
        model = train_euclidean_distance(top5_roc, data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['roc_euclidean'] = test_euclidean_distance(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_mahalanobis_distance(top5_roc, data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['roc_mahalanobis'] = test_mahalanobis_distance(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_fisher_lda(top5_roc, data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['roc_fisher'] = test_fisher_lda(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

    # --- Kruskal-Wallis selected features ---
    if top5_kruskall is not None:
        model = train_euclidean_distance(top5_kruskall, data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['kw_euclidean'] = test_euclidean_distance(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_mahalanobis_distance(top5_kruskall, data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['kw_mahalanobis'] = test_mahalanobis_distance(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_fisher_lda(top5_kruskall, data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['kw_fisher'] = test_fisher_lda(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

    # --- PCA features ---
    if X_pca_train is not None and X_pca_test is not None:
        df_pca_train = pd.DataFrame(X_pca_train, columns=[f"PC{i+1}" for i in range(X_pca_train.shape[1])])
        df_pca_test = pd.DataFrame(X_pca_test, columns=[f"PC{i+1}" for i in range(X_pca_test.shape[1])])
        pca_features = list(df_pca_train.columns)

        model = train_euclidean_distance(pca_features, data=df_pca_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['pca_euclidean'] = test_euclidean_distance(model, data=df_pca_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_mahalanobis_distance(pca_features, data=df_pca_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['pca_mahalanobis'] = test_mahalanobis_distance(model, data=df_pca_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_fisher_lda(pca_features, data=df_pca_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['pca_fisher'] = test_fisher_lda(model, data=df_pca_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

    # --- LDA first component (1D) ---
    if LD1_train is not None and LD1_test is not None:
        df_lda_train = pd.DataFrame(LD1_train, columns=['LD1'])
        df_lda_test = pd.DataFrame(LD1_test, columns=['LD1'])

        model = train_euclidean_distance(['LD1'], data=df_lda_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['lda_euclidean'] = test_euclidean_distance(model, data=df_lda_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_mahalanobis_distance(['LD1'], data=df_lda_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['lda_mahalanobis'] = test_mahalanobis_distance(model, data=df_lda_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_fisher_lda(['LD1'], data=df_lda_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['lda_fisher'] = test_fisher_lda(model, data=df_lda_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

    return results

