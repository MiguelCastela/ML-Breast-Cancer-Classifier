
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



def knn_classifier_search(X_train, y_train, X_test, y_test, k=1, plot=False, feature_names=None):
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

    runs=np.arange(0,30)
    ks=np.arange(1,30,2)
    errMat=np.zeros((runs.shape[0],ks.shape[0]))

    for r in runs:
        ki = 0
        for k in ks:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            hits = np.sum(y_pred == y_test)
            accuracy = hits / y_test.shape[0]
            erro_percent = (1 - accuracy) * 100

            errMat[r,ki]=erro_percent
            ki+=1
    avgErr=np.mean(errMat,axis=0)
    stdErr=np.std(errMat,axis=0)

    """
    fig = px.scatter(x=ks, y=avgError,error_y=stdError)
    fig.update_layout(
        xaxis=dict(
            title=dict(
                text="k"
            )
        ),
        yaxis=dict(
            title=dict(
                text="Average Erros±Std"
            )
        ),
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple"
        )
    )
    fig.show()
    """

    optK = ks[np.where(avgErr==np.min(avgErr))[0]][0]
    print("best k=" + str(optK))
    


    knn1 = KNeighborsClassifier(n_neighbors=optK)
    knn1.fit(X_train, y_train)
    y_pred = knn1.predict(X_test)

    xlabel = feature_names[0] if feature_names else "Feature 1"
    ylabel = feature_names[1] if feature_names else "Feature 2"

    Hits=(np.where(y_pred==y_test)[0]).shape[0]

    Acc=Hits/y_test.shape[0]

    Err=(1-Acc)*100

    print("Error (%)="+str(Err))

    
    disp = DecisionBoundaryDisplay.from_estimator(
            knn1, X_test, response_method="predict",
            xlabel=xlabel, ylabel=ylabel, alpha=0.5,
        )
    
    #disp.ax_.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, edgecolor="k")

    return y_pred