from sklearn import svm
import numpy as np
from sklearn.neighbors import KNeighborsClassifier



def knn_classifier_search(X_train, y_train, X_val, y_val, ks=np.arange(1, 30, 2)):
    """
    Finds the optimal K for KNN using robust multi-run evaluation on validation data (X_val, y_val).
    X_train, y_train: Training data (NumPy arrays)
    X_val, y_val: Validation data (NumPy arrays)
    Returns: best_K (int)
    """

    # Use the original multi-run structure for robust error estimation
    runs = np.arange(0, 30)
    ks = np.arange(1, 30, 2) # Default list of K values (overridden if passed in)
    errMat = np.zeros((runs.shape[0], ks.shape[0]))

    for r in runs:
        ki = 0
        for k in ks:
            # 1. Fit classifier on training data
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            
            # 2. Predict on validation data
            y_pred_val = knn.predict(X_val)
            
            # 3. Calculate error (using accuracy and converting to error percentage)
            accuracy = np.sum(y_pred_val == y_val) / y_val.shape[0]
            erro_percent = (1 - accuracy) * 100

            errMat[r, ki] = erro_percent
            ki += 1
            
    # Compute average error over all runs/folds
    avgErr = np.mean(errMat, axis=0)

    # Find the optimal K that minimizes the average error
    optK = ks[np.where(avgErr == np.min(avgErr))[0]][0]
    
    print(f"KNN Optimal K found: {optK} (Average Val Error: {np.min(avgErr):.4f}%)")
    
    # Only return the optimal hyperparameter value
    return optK


def svm_search(X_train, y_train, X_val, y_val, CsP2=np.arange(-5.0, 10.0, 1.0)):
    """
    Finds optimal C for Linear SVM using robust multi-run evaluation on validation data.
    Returns: best_C (float)
    """
    
    # Use the original multi-run structure for robust error estimation
    runs = np.arange(0, 10) # Assuming 10 runs as typical in search examples
    C_list = 2**CsP2
    errMat = np.zeros((runs.shape[0], C_list.shape[0]))
    
    for r in runs:
        ki = 0
        for C in C_list:
            # 1. Fit classifier on training data
            clf = svm.SVC(kernel="linear", C=C)
            clf.fit(X_train, y_train)
            
            # 2. Predict on validation data
            y_pred_val = clf.predict(X_val)
            
            # 3. Calculate error
            accuracy = np.sum(y_pred_val == y_val) / y_val.shape[0]
            errMat[r, ki] = (1 - accuracy) * 100
            ki += 1
            
    avgErr = np.mean(errMat, axis=0)
    
    # Find the optimal C that minimizes the average error
    optCP2 = CsP2[np.where(avgErr == np.min(avgErr))[0]][0]
    best_C = 2**optCP2
            
    print(f"Linear SVM Optimal C (2^{optCP2}) found: {best_C:.6f} (Average Val Error: {np.min(avgErr):.4f}%)")
    return best_C
    


def svm_rbf_kernel_search(X_train, y_train, X_val, y_val, 
                          CsP2=np.arange(-5.0, 10.0, 1.0), 
                          GsP2=np.arange(-10.0, 5.0, 1.0)):
    """
    Finds optimal C and Gamma for RBF SVM using robust multi-run evaluation on validation data.
    Returns: best_C (float), best_Gamma (float)
    """
    
    # Use the original multi-run structure for robust error estimation
    runs = np.arange(0, 10) # Assuming 10 runs
    C_list = 2**CsP2
    Gamma_list = 2**GsP2
    
    # Error matrix: C x Gamma x Runs
    errMat = np.zeros((C_list.shape[0], Gamma_list.shape[0], runs.shape[0]))
    
    for r in runs:
        ki = 0
        for C in C_list:
            ui = 0
            for Gamma in Gamma_list:
                # 1. Fit classifier on training data
                clf = svm.SVC(kernel="rbf", C=C, gamma=Gamma)
                clf.fit(X_train, y_train)
                
                # 2. Predict on validation data
                y_pred_val = clf.predict(X_val)
                
                # 3. Calculate error
                accuracy = np.sum(y_pred_val == y_val) / y_val.shape[0]
                errMat[ki, ui, r] = (1 - accuracy) * 100
                ui += 1
            ki += 1
            
    # Compute average error over all runs
    avgError = np.mean(errMat, axis=2)
    minErrorIx = np.where(avgError == avgError.min())
    
    # Find optimal C and Gamma
    optCP2 = CsP2[minErrorIx[0]][0]
    optGP2 = GsP2[minErrorIx[1]][0]
    best_C = 2**optCP2
    best_Gamma = 2**optGP2
    
    print(f"RBF SVM Optimal C (2^{optCP2}): {best_C:.6f}, Optimal Gamma (2^{optGP2}): {best_Gamma:.6f} (Avg Val Error: {avgError.min():.4f}%)")
    return best_C, best_Gamma

