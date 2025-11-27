from xml.parsers.expat import model
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_curve, auc
import numpy as np
import scipy.io as sio
import plotly.express as px
import plotly.graph_objects as go
from sklearn import mixture
from scipy.stats import multivariate_normal
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt



def train_euclidean_distance(feature_names, method_name="Unknown", data=None, ixHealthy=None, ixCancer=None):

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
    dist_healthy = np.zeros(len(X))
    dist_cancer = np.zeros(len(X))
    y_pred = np.zeros(len(X))
    for i, sample in enumerate(X):
        dist_healthy[i] = np.linalg.norm(sample - mu_healthy)
        dist_cancer[i] = np.linalg.norm(sample - mu_cancer)
        y_pred[i] = 0 if dist_healthy[i] < dist_cancer[i] else 1
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

    decision_scores = dist_healthy - dist_cancer

    fpr, tpr, _ = roc_curve(y_true, decision_scores)
    roc_auc = auc(fpr, tpr)


    print(f"Sensitivity (%) = {sensitivity * 100:.2f}")
    print(f"Specificity (%) = {specificity * 100:.2f}")
    print(f"Precision (%) = {precision * 100:.2f}")
    print(f"F1 Score (%) = {f1_score * 100:.2f}")
    print(f"Accuracy (%) = {accuracy * 100:.2f}")
    print(f"ROC-AUC (%) = {roc_auc * 100:.2f}")
    return accuracy, sensitivity, specificity, precision, f1_score, roc_auc


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


def train_fisher_lda(feature_names, method_name="Unknown", data=None , ixHealthy=None, ixCancer=None):

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


    decision_values = (X @ w).flatten() + b
    y_pred = np.zeros(len(X))
    y_pred[decision_values > 0] = 1  # Cancer class
    
    TP = np.sum((y_true == 1) & (y_pred == 1))  # Cancer correctly classified
    TN = np.sum((y_true == 0) & (y_pred == 0))  # Healthy correctly classified
    FP = np.sum((y_true == 0) & (y_pred == 1))  # Healthy misclassified as Cancer
    FN = np.sum((y_true == 1) & (y_pred == 0))  # Cancer misclassified as Healthy
    
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    fpr, tpr, _ = roc_curve(y_true, decision_values)
    roc_auc = auc(fpr, tpr)

    
    print(f"Sensitivity (%) = {sensitivity * 100:.2f}")
    print(f"Specificity (%) = {specificity * 100:.2f}")
    print(f"Precision (%) = {precision * 100:.2f}")
    print(f"F1 Score (%) = {f1_score * 100:.2f}")
    print(f"Accuracy (%) = {accuracy * 100:.2f}")
    print(f"ROC-AUC (%) = {roc_auc * 100:.2f}")

    return accuracy, sensitivity, specificity, precision, f1_score, roc_auc


def bayes_fit_predict(Xtr, ytr, Xte, return_model=False, reg=1e-6):
    """
    Classificador Bayesiano Gaussiano (2 classes) numa única função.
    - Xtr: ndarray [n_train, d]
    - ytr: rótulos de treino (ex.: {1,2} ou {0,1})
    - Xte: ndarray [n_test, d]
    - return_model: se True, devolve também o modelo (médias, covariâncias e priors)
    - reg: regularização adicionada à diagonal das covariâncias
    Retorna: y_pred (mesma codificação de ytr) e, opcionalmente, o modelo.
    """
    ytr = np.asarray(ytr).squeeze()
    classes = np.unique(ytr)
    if classes.size != 2:
        raise ValueError("Este classificador suporta apenas 2 classes.")

    c1, c2 = classes[0], classes[1]
    ix1 = np.where(ytr == c1)[0]
    ix2 = np.where(ytr == c2)[0]

    Pw1=ix1.shape[0]/(ix1.shape[0]+ix2.shape[0])
    Pw2=ix2.shape[0]/(ix1.shape[0]+ix2.shape[0])

    clf1 = mixture.GaussianMixture(n_components=1)
    clf2 = mixture.GaussianMixture(n_components=1)
    mod1=clf1.fit(Xtr[ix1,:])
    mod2=clf2.fit(Xtr[ix2,:])

    mean1 = mod1.means_.squeeze()
    mean2 = mod2.means_.squeeze()

    cov1 = mod1.covariances_[0]
    cov2 = mod2.covariances_[0]

    # Regularização para evitar matrizes singulares
    if cov1.ndim == 0:
        cov1 = np.array([[cov1 + reg]])
        cov2 = np.array([[cov2 + reg]])
    else:
        cov1 = cov1 + reg * np.eye(cov1.shape[0])
        cov2 = cov2 + reg * np.eye(cov2.shape[0])

    # Log-posteriors para maior estabilidade numérica
    logp1 = multivariate_normal.logpdf(Xte, mean=mean1, cov=cov1) + np.log(Pw1)
    logp2 = multivariate_normal.logpdf(Xte, mean=mean2, cov=cov2) + np.log(Pw2)

    y_pred = np.where(logp1 >= logp2, c1, c2)

    if return_model:
        model = {'mean1': mean1, 'mean2': mean2,
                 'cov1': cov1, 'cov2': cov2,
                 'Pw1': Pw1, 'Pw2': Pw2,
                 'classes': (c1, c2)}
        return y_pred, model
    return y_pred

# ...existing code...
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt

def knn_classificar(X_train, y_train, X_test, y_test=None, k=1, plot=False, feature_names=None):
    """
    Classificador k-NN simples.
    X_train, X_test: arrays (n_samples, n_features)
    y_train: rótulos de treino
    y_test: rótulos verdadeiros (opcional) para métricas
    k: número de vizinhos
    plot: se True e n_features==2, desenha fronteira de decisão
    feature_names: lista com nomes das 2 features (apenas para labels do gráfico)
    Retorna: y_pred, accuracy, erro_percent
    """
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = None
    erro_percent = None
    if y_test is not None:
        hits = np.sum(y_pred == y_test)
        accuracy = hits / y_test.shape[0]
        erro_percent = (1 - accuracy) * 100
        print(f"k={k}  Error (%)={erro_percent:.2f}")

    if plot and X_test.shape[1] == 2:
        xlabel = feature_names[0] if feature_names else "Feature 1"
        ylabel = feature_names[1] if feature_names else "Feature 2"
        disp = DecisionBoundaryDisplay.from_estimator(
            clf, X_test, response_method="predict",
            xlabel=xlabel, ylabel=ylabel, alpha=0.5,
        )
        disp.ax_.scatter(X_test[:, 0], X_test[:, 1], c=y_pred if y_test is None else y_test, edgecolor="k")
        plt.title(f"k-NN (k={k})")
        plt.show()

    return y_pred, accuracy, erro_percent


def knn_classifier(X_train, y_train, X_test, y_test=None, k=1, plot=False, feature_names=None):
    """
    Classificador k-NN simples.
    X_train, X_test: arrays (n_samples, n_features)
    y_train: rótulos de treino
    y_test: rótulos verdadeiros (opcional) para métricas
    k: número de vizinhos
    plot: se True e n_features==2, desenha fronteira de decisão
    feature_names: lista com nomes das 2 features (apenas para labels do gráfico)
    Retorna: y_pred, accuracy, erro_percent
    """
    knn1 = KNeighborsClassifier(n_neighbors=k)
    knn1.fit(X_train, y_train)
    y_pred = knn1.predict(X_test)

    accuracy = None
    erro_percent = None
    if y_test is not None:
        hits = np.sum(y_pred == y_test)
        accuracy = hits / y_test.shape[0]
        erro_percent = (1 - accuracy) * 100
        print(f"k={k}  Error (%)={erro_percent:.2f}")

    if plot and X_test.shape[1] == 2:
        xlabel = feature_names[0] if feature_names else "Feature 1"
        ylabel = feature_names[1] if feature_names else "Feature 2"
        disp = DecisionBoundaryDisplay.from_estimator(
            knn1, X_test, response_method="predict",
            xlabel=xlabel, ylabel=ylabel, alpha=0.5,
        )
        disp.ax_.scatter(X_test[:, 0], X_test[:, 1], c=y_pred if y_test is None else y_test, edgecolor="k")
        plt.title(f"k-NN (k={k})")
        plt.show()

    return y_pred, accuracy, erro_percent


def run_all_classifiers(train_data, test_data, ixHealthy_train, ixCancer_train, ixHealthy_test, ixCancer_test, all_features, top5_roc=None, top5_kruskall=None, X_pca_train=None, X_pca_test=None, LD1_train=None, LD1_test=None):
   

    """
    Run classifiers using different feature sets and train/test split:
    - train_data / test_data: pandas DataFrames with the same columns
    - top5_roc / top5_kruskall: lists of column names from original dataframe
    - X_pca_train / X_pca_test: numpy arrays for PCA features
    - LD1_train / LD1_test: numpy arrays for LDA1 (1D) features
    """
    results = {}




    model = train_euclidean_distance(all_features, method_name="All Features", 
                                    data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
    results['all_euclidean'] = test_euclidean_distance(model, data=test_data, 
                                                    ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

    model = train_mahalanobis_distance(all_features, method_name="All Features", 
                                    data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
    results['all_mahalanobis'] = test_mahalanobis_distance(model, data=test_data, 
                                                        ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

    model = train_fisher_lda(all_features, method_name="All Features", 
                            data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
    results['all_fisher'] = test_fisher_lda(model, data=test_data, 
                                            ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)


    
    if top5_roc is not None:
        model = train_euclidean_distance(top5_roc, method_name="ROC-AUC", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['roc_euclidean'] = test_euclidean_distance(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_mahalanobis_distance(top5_roc, method_name="ROC-AUC", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['roc_mahalanobis'] = test_mahalanobis_distance(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_fisher_lda(top5_roc, method_name="ROC-AUC", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['roc_fisher'] = test_fisher_lda(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

    if top5_kruskall is not None:
        model = train_euclidean_distance(top5_kruskall, method_name="Kruskal-Wallis", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['kw_euclidean'] = test_euclidean_distance(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_mahalanobis_distance(top5_kruskall, method_name="Kruskal-Wallis", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['kw_mahalanobis'] = test_mahalanobis_distance(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_fisher_lda(top5_kruskall, method_name="Kruskal-Wallis", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['kw_fisher'] = test_fisher_lda(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

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

    if LD1_train is not None and LD1_test is not None:


        df_lda_train = pd.DataFrame(LD1_train, columns=['LD1'])
        df_lda_test = pd.DataFrame(LD1_test, columns=['LD1'])
    

        model = train_euclidean_distance(['LD1'], method_name="LDA", data=df_lda_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['lda_euclidean'] = test_euclidean_distance(model, data=df_lda_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_mahalanobis_distance(['LD1'], method_name="LDA", data=df_lda_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['lda_mahalanobis'] = test_mahalanobis_distance(model, data=df_lda_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

    return results


def run_classifiers_multiple_times(train_data, test_data, ixHealthy_train, ixCancer_train, ixHealthy_test, ixCancer_test,
                                   all_features, top5_roc=None, top5_kruskall=None, X_pca_train=None, X_pca_test=None,
                                   LD1_train=None, LD1_test=None, n_runs=5):
    """
    Run run_all_classifiers n_runs times and compute mean metrics.
    """
    all_results = {}

    for run in range(n_runs):
        print(f"\n--- Run {run + 1} ---")
        results = run_all_classifiers(
            train_data=train_data,
            test_data=test_data,
            ixHealthy_train=ixHealthy_train,
            ixCancer_train=ixCancer_train,
            ixHealthy_test=ixHealthy_test,
            ixCancer_test=ixCancer_test,
            all_features=all_features,
            top5_roc=top5_roc,
            top5_kruskall=top5_kruskall,
            X_pca_train=X_pca_train,
            X_pca_test=X_pca_test,
            LD1_train=LD1_train,
            LD1_test=LD1_test
        )

        for key, metrics in results.items():
            if key not in all_results:
                all_results[key] = []
            all_results[key].append(metrics)

    # Compute mean metrics
    mean_results = {k: tuple(np.mean(v, axis=0)) for k, v in all_results.items()}

    # Print mean results
    print("\n=== Average metrics over", n_runs, "runs ===")
    for clf, metrics in mean_results.items():
        print(f"{clf}: Accuracy={metrics[0]:.3f}, Sensitivity={metrics[1]:.3f}, Specificity={metrics[2]:.3f}, "
              f"Precision={metrics[3]:.3f}, F1={metrics[4]:.3f}, ROC-AUC={metrics[5]:.3f}")

    return mean_results
    


          
                                 

