
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
import numpy as np
from sklearn import svm


# Helper function to convert 0/1 labels to -1/+1
def map_labels_to_pm1(y):
    # Converts 0 (Healthy) -> -1 and 1 (Cancer) -> 1
    return np.where(y == 1, 1, -1)

# Helper function to convert -1/+1 labels back to 0/1
def map_labels_to_01(y):
    # Converts -1 -> 0 and 1 -> 1
    return np.where(y == 1, 1, 0)


def train_svm_custom(feature_names, method_name="Unknown", data=None, ixHealthy=None, ixCancer=None, C=1e10, tol=1e-3, max_passes=100):
    """
    Trains a Linear SVM using the provided custom SMO implementation (svm_hard_margin logic).
    C is set to a large value (1e10) for hard-margin, matching the default in the original code.
    """
    if ixHealthy is None or ixCancer is None:
        ixHealthy = np.where(data["Classification"] == "Healthy")
        ixCancer = np.where(data["Classification"] == "Cancer")

    print(f"\n=== CUSTOM LINEAR SVM CLASSIFIER (C={C}, {method_name}) ===")
    print(f"Using features: {feature_names}")

    # --- Data Extraction and Label Conversion ---
    X_train = data[feature_names].iloc[ixHealthy[0]].values # Healthy
    X_train = np.concatenate([X_train, data[feature_names].iloc[ixCancer[0]].values], axis=0)
    
    # y_train: 0/1 labels from train indices
    y_train_01 = np.concatenate([np.zeros(len(ixHealthy[0])), np.ones(len(ixCancer[0]))])
    # Convert to -1/+1 for the SMO algorithm
    y_train_pm1 = map_labels_to_pm1(y_train_01) 


    # --------------------------------------------------------------------------------------
    # START: Code adapted directly from svm_hard_margin (with label change: y_train_pm1)
    
    X = np.asmatrix(X_train)
    y = np.asmatrix(y_train_pm1).T
    m, n = X.shape

    # Inner helper functions for SMO (keeping them outside to avoid copy-pasting their body)
    # NOTE: You must ensure clipAlphasJ and selectJrandom are available in the scope 
    # (e.g., defined outside both train/test functions or copied here).
    
    # Since they were defined *inside* the original function, we'll define them here:
    def clipAlphasJ(aj, H, L):
        if aj > H: aj = H
        if L > aj: aj = L
        return aj

    def selectJrandom(i, m):
        j = i
        while j == i:
            j = int(np.random.uniform(0, m))
        return j


    b = 0.0
    alphas = np.asmatrix(np.zeros((m, 1)))
    passes = 0

    while passes < max_passes:
        num_changed_alphas = 0
        for i in range(m):
            # The calculation of fXi uses the *current* alphas and b
            fXi = float(np.multiply(alphas, y).T * (X * X[i, :].T)) + b
            Ei = fXi - float(y[i])
            if ((y[i] * Ei < -tol) and (alphas[i] < C)) or ((y[i] * Ei > tol) and (alphas[i] > 0)):
                j = selectJrandom(i, m)
                fXj = float(np.multiply(alphas, y).T * (X * X[j, :].T)) + b
                Ej = fXj - float(y[j])

                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()

                if (y[i] != y[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])

                if L == H:
                    continue

                eta = 2.0 * X[i, :] * X[j, :].T - X[i, :] * X[i, :].T - X[j, :] * X[j, :].T
                if eta >= 0:
                    continue

                alphas[j] -= y[j] * (Ei - Ej) / eta
                alphas[j] = clipAlphasJ(alphas[j], H, L)

                if abs(alphas[j] - alphaJold) < 1e-5:
                    continue

                alphas[i] += y[j] * y[i] * (alphaJold - alphas[j])

                b1 = b - Ei - y[i] * (alphas[i] - alphaIold) * X[i, :] * X[i, :].T - y[j] * (alphas[j] - alphaJold) * X[i, :] * X[j, :].T
                b2 = b - Ej - y[i] * (alphas[i] - alphaIold) * X[i, :] * X[j, :].T - y[j] * (alphas[j] - alphaJold) * X[j, :] * X[j, :].T

                if (0 < alphas[i]) and (C > alphas[i]):
                    b = float(b1)
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = float(b2)
                else:
                    b = float((b1 + b2) / 2.0)

                num_changed_alphas += 1

        passes = passes + 1 if num_changed_alphas == 0 else 0

    # Compute w
    w = np.zeros((1, n))
    y_vec = np.asarray(y).flatten()
    for i in range(m):
        w += float(alphas[i]) * y_vec[i] * np.asarray(X[i, :])
    w = np.asmatrix(w).T  # shape (n,1)

    # Support vectors
    #Identify samples that are support vectors
    ixSv = np.where(np.asarray(alphas).flatten() > 0)[0]


    Svs = np.asarray(X)[ixSv, :]
    ySvs = y_vec[ixSv]

    # Compute b via média de dois SVs (se existirem de ambos os lados)
    if np.any(ySvs == 1) and np.any(ySvs == -1):
        #Identify the positive and negative support vectors
        pos_sv = Svs[ySvs == 1][0]
        neg_sv = Svs[ySvs == -1][0]
        b_alt = -0.5 * (float(w.T @ np.asmatrix(pos_sv).T) + float(w.T @ np.asmatrix(neg_sv).T))
        b = b_alt
        
    # END: Code adapted from svm_hard_margin
    # --------------------------------------------------------------------------------------


    # Define the predict function based on the calculated w and b
    def predict_scores(Xte):
        # Xte must be a matrix or array suitable for matrix multiplication
        Xte = np.asmatrix(Xte) if isinstance(Xte, np.ndarray) else Xte
        return (Xte * w + b).A.flatten() # Scores/Distance to hyperplane

    def predict_labels(Xte):
        scores = predict_scores(Xte)
        # Prediction in {-1, 1}
        y_pred_pm1 = np.where(scores >= 0, 1, -1)
        # Convert back to {0, 1} for compatibility with run_all_classifiers
        return map_labels_to_01(y_pred_pm1)


    print(f"SVM weights (w): {w.A.flatten()}")
    print(f"SVM bias (b): {b}")

    # Store the essential parts and prediction functions
    return {
        "feature_names": feature_names,
        "w": w,
        "b": b,
        "predict_labels": predict_labels,  # Returns 0/1 labels
        "predict_scores": predict_scores,  # Returns raw scores (distance to hyperplane)
    }

def test_svm_custom(model, data=None , ixHealthy=None, ixCancer=None):
    """
    Classifies test data using the custom linear SVM model and computes metrics.
    """
    if ixHealthy is None or ixCancer is None:
        ixHealthy = np.where(data["Classification"] == "Healthy")
        ixCancer = np.where(data["Classification"] == "Cancer")

    feature_names = model["feature_names"]
    
    # --- Test Data Preparation ---
    X_test = data[feature_names].iloc[ixHealthy[0]].values # Healthy
    X_test = np.concatenate([X_test, data[feature_names].iloc[ixCancer[0]].values], axis=0)

    # y_true is the ground truth in 0/1
    y_true = np.concatenate([np.zeros(len(ixHealthy[0])), np.ones(len(ixCancer[0]))])  # 0=Healthy, 1=Cancer

    # --- Prediction and Decision Scores ---
    y_pred = model["predict_labels"](X_test)
    decision_scores = model["predict_scores"](X_test) # Raw scores for ROC-AUC

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



def train_svm_linear(feature_names, method_name="Unknown", data=None, ixHealthy=None, ixCancer=None, C=1):
    """
    Trains a Linear Support Vector Machine (SVM) classifier.
    """
    if ixHealthy is None or ixCancer is None:
        ixHealthy = np.where(data["Classification"] == "Healthy")
        ixCancer = np.where(data["Classification"] == "Cancer")

    print(f"\n=== LINEAR SVM CLASSIFIER (C={C}, {method_name}) ===")
    print(f"Using features: {feature_names}")

    # Extract training data
    X_train = data[feature_names].values
    # Target labels must be 0/1 for consistency
    y_train = np.concatenate([np.zeros(len(ixHealthy[0])), np.ones(len(ixCancer[0]))]) 

    # Fit SVM model with linear kernel
    # probability=True allows predict_proba for ROC-AUC, though decision_function is often preferred for SVM scores.
    clf = svm.SVC(kernel="linear", C=C, random_state=42, probability=True) 
    clf.fit(X_train, y_train)

    # Note: SVM stores the decision boundary as w and b similar to your custom code, 
    # but we store the whole object for simplicity.
    w = clf.coef_.flatten()
    b = clf.intercept_[0]
    
    print(f"SVM weights: {w}")
    print(f"SVM bias: {b}")

    model = {
        "feature_names": feature_names,
        "clf": clf,
        "C": C,
    }

    return model





def train_svm_rbf_kernel(feature_names, method_name="Unknown", data=None, ixHealthy=None, ixCancer=None, C=2**5.0, gamma=2**-15.0):
    """
    Trains an RBF Kernel Support Vector Machine (SVM) classifier using pre-determined optimal C and gamma.
    """
    if ixHealthy is None or ixCancer is None:
        ixHealthy = np.where(data["Classification"] == "Healthy")
        ixCancer = np.where(data["Classification"] == "Cancer")

    print(f"\n=== RBF-KERNEL SVM CLASSIFIER (C={C:.4f}, Gamma={gamma:.6f}, {method_name}) ===")
    print(f"Using features: {feature_names}")

    # Extract training data
    X_train = data[feature_names].values
    y_train = np.concatenate([np.zeros(len(ixHealthy[0])), np.ones(len(ixCancer[0]))]) # 0=Healthy, 1=Cancer

    # Fit RBF SVM model
    # probability=True is necessary to get predict_proba for ROC-AUC if needed, 
    # although decision_function is generally used for scores.
    clf = svm.SVC(kernel="rbf", C=C, gamma=gamma, random_state=42, probability=True) 
    clf.fit(X_train, y_train)

    model = {
        "feature_names": feature_names,
        "clf": clf,
        "C": C,
        "gamma": gamma
    }

    return model


def test_svm(model, data=None , ixHealthy=None, ixCancer=None):
    """
    Classifies test data using the trained Linear SVM model and computes metrics.
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

    # Decision scores for SVM is the distance to the hyperplane (w*x + b). 
    # Higher score favors class 1 (Cancer).
    decision_scores = clf.decision_function(X_test) 

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


def svm_search(X_train, y_train, X_test, y_test, runs = np.arange(0,10), CsP2 = np.arange(-20.0,20.0,2.0)):
    """
    Realiza busca em grid para encontrar melhor hiperparâmetro C do SVM hard-margin.
    - X_train, y_train: dados de treino
    - X_test, y_test: dados de teste
    - runs: array com índices de múltiplas execuções
    - CsP2: array com potências de 2 para testar C=2^CsP2
    Retorna: melhor C e matriz de erros (runs x CsP2)
    """

    errMat=np.zeros((runs.shape[0],CsP2.shape[0]))

    for r in runs:#Create several training/testing partitions
        ki=0
        for cP2 in CsP2:#Evaluate different Cs
            #fit SVM
            clf = svm.SVC(kernel="linear", C=2**cP2)
            clf.fit(X_train, y_train)
            yp=clf.predict(X_test)
            #Evaluate error
            Hits=(np.where(yp==y_test)[0]).shape[0]
            Acc=Hits/y_test.shape[0]
            
            #Store error
            errMat[r,ki]=(1-Acc)*100
            ki=ki+1
            
    #at the end compute average error and error standard deviation
    avgError=np.mean(errMat,axis=0)
    stdError=np.std(errMat,axis=0)
    xTickTexts=[]
    for cP2 in CsP2:
        xTickTexts.append("2^"+str(int(cP2)))


    """
    #Display average error as function of power of C
    fig = px.scatter(x=CsP2, y=avgError,error_y=stdError)
    fig.update_layout(
        xaxis=dict(
            title=dict(text="C"),
            tickmode = 'array',
            tickvals = CsP2,
            ticktext = xTickTexts
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

    #Fing the best C
    optC=CsP2[np.where(avgError==np.min(avgError))[0]][0]
    print("Best C=2^"+str(optC))

    #Train again SVM with best C
    clf = svm.SVC(kernel="linear", C=2**optC)
    clf.fit(X_train, y_train)

    yp=clf.predict(X_test)

    Hits=(np.where(yp==y_test)[0]).shape[0]

    Acc=Hits/y_test.shape[0]

    Err=(1-Acc)*100

    print("Error (%)="+str(Err))

    


def svm_rbf_kernel_search(X_train, y_train, X_test, y_test, runs=np.arange(0,10), CsP2=np.arange(-10,25.0,1.0), GsP2=np.arange(-25.0,0.0,1.0)):

    errMat=np.zeros((CsP2.shape[0],GsP2.shape[0],runs.shape[0]))

    for r in runs:#Create several training/testing partitions
        ki=0
        for cP2 in CsP2:#Evaluate different Cs
            ui=0
            for gP2 in GsP2:#Evaluate different Gammas
                #fit SVM
                clf = svm.SVC(kernel="rbf", C=2**cP2,gamma=2**gP2)
                clf.fit(X_train, y_train)
                yp=clf.predict(X_test)
                #Evaluate error
                Hits=(np.where(yp==y_test)[0]).shape[0]
                Acc=Hits/y_test.shape[0]

                #Store error
                errMat[ki,ui,r]=(1-Acc)*100
                ui=ui+1
            ki=ki+1
            
    #at the end compute average error 
    avgError=np.mean(errMat,axis=2)
    minErrorIx=np.where(avgError==avgError.min())


    #Plot Error 
    yTickTexts=[]
    for cP2 in CsP2:
        yTickTexts.append("2^"+str(int(cP2)))
        
    xTickTexts=[]
    for gP2 in GsP2:
        xTickTexts.append("2^"+str(int(gP2)))


    """
    fig=go.Figure()                             

    fig.add_trace(go.Heatmap(
            z=avgError,
            x=GsP2,
            y=CsP2,
            colorscale='Viridis'))



    fig.add_scatter(x=GsP2[minErrorIx[1]],y=CsP2[minErrorIx[0]],mode='markers',marker_size=15,
                    marker_symbol="x",marker=dict(color="white"),line=dict(width=8,color='white'),name="Minimum error")
    #fig.add_trace(go.Surface(x=Xp,y=Yp,z=Z2))
    #fig.update_traces(contours_z=dict(show=True, usecolormap=True,highlightcolor="limegreen", project_z=True))

    #fig.update_xaxes(title_text="Gamma")
    #fig.update_yaxes(title_text="C")


    fig.update_layout(
        xaxis=dict(
            title=dict(text="Gamma"),
            tickmode = 'array',
            tickvals = GsP2,
            ticktext = xTickTexts
        ),
        yaxis=dict(
            title=dict(text="C"),
            tickmode = 'array',
            tickvals = CsP2,
            ticktext = yTickTexts),
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple"
        ),
        autosize=False,
        width=900,
        height=800)


    fig.show()
    """

    print("One Best C=")
    print("2^"+str(int(CsP2[minErrorIx[0]][0])))

    print("One Best Gamma=")
    print("2^"+str(int(GsP2[minErrorIx[1]][0])))

    bestCP2=CsP2[minErrorIx[0]][0]
    bestGP2=GsP2[minErrorIx[1]][0]
    clf = svm.SVC(kernel="rbf", C=2**bestCP2,gamma=2**bestGP2)
    clf.fit(X_train, y_train)

    """
    # Plotting settings
    fig, ax = plt.subplots(figsize=(7, 7))
    x_min, x_max, y_min, y_max = np.min(X_train[:,0]),np.max(X_train[:,0]),np.min(X_train[:,1]),np.max(X_train[:,1])
    ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))

    # Plot samples by color and add legend
    scatter = ax.scatter(X_train[:, 0], X_train[:, 1], s=50, c=y_train, label=y_train, edgecolors="k")
    plt.xlabel("N")
    plt.ylabel("PRT/10")

    # Plot decision boundary and margins
    common_params = {"estimator": clf, "X": X, "ax": ax}
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="predict",
        plot_method="pcolormesh",
        alpha=0.3,
    )
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="decision_function",
        plot_method="contour",
        levels=[-1, 0, 1],
        colors=["k", "k", "k"],
        linestyles=["--", "-", "--"],
    )

    # Plot bigger circles around samples that serve as support vectors
    ax.scatter(
        clf.support_vectors_[:, 0],
        clf.support_vectors_[:, 1],
        s=150,
        facecolors="none",
        edgecolors="k",
        )

    _ = plt.show()
    """


