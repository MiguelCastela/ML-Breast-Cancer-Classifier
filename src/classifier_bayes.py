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
    """
    Trains Bayesian classifier by estimating Gaussian parameters for each class.
    """

    print(f"\n=== BAYESIAN CLASSIFIER ({method_name}) ===")
    print(f"Using features: {feature_names}")

    # --- Data Extraction and Labeling ---
    X_train = data[feature_names].values
    # Use 0 and 1 for labels, consistent with test functions
    y_train = np.concatenate([np.zeros(len(ixHealthy[0])), np.ones(len(ixCancer[0]))])
    
    # Indices for the two classes (0=Healthy, 1=Cancer)
    ix0 = np.where(y_train == 0)[0]
    ix1 = np.where(y_train == 1)[0]


    Pw0=ix0.shape[0]/(ix0.shape[0]+ix1.shape[0])
    Pw1=ix1.shape[0]/(ix0.shape[0]+ix1.shape[0])

    #compute priors

    # Estimate gaussian conditional PDFs --> fit two Gaussian Mixture Models

    clf1 = mixture.GaussianMixture(n_components=1)
    clf2 = mixture.GaussianMixture(n_components=1)
    mod1=clf1.fit(X_train[ix0,:])
    mod2=clf2.fit(X_train[ix1,:])

    mean1 = mod1.means_.squeeze()
    mean2 = mod2.means_.squeeze()

    cov1 = mod1.covariances_[0]
    cov2 = mod2.covariances_[0]

    # Regularização para evitar matrizes singulares, explicar isto no relatorio

    reg = 1e-6
    if cov1.ndim == 0:
        cov1 = np.array([[cov1 + reg]])
        cov2 = np.array([[cov2 + reg]])
    else:
        cov1 = cov1 + reg * np.eye(cov1.shape[0])
        cov2 = cov2 + reg * np.eye(cov2.shape[0])



    model = {
        "feature_names": feature_names,
        "classes": (0, 1)  # 0=Healthy, 1=Cancer
    }

        #NAO é aqui que se usaria a função pdfGauss e usebayes do stor? onde meter o testBayes classifier?

    """
    def pdfGauss(X,mean,cov):
        covInv=np.linalg.inv(cov)
        dim=cov.shape[0]
        val=np.array([])
        for i in range(X.shape[0]):
            dist=((np.array([X[i,:]-mean]))@covInv@(np.array([X[i,:]-mean])).T).squeeze()
            multivariate_pdf = np.exp(-0.5*dist)/((2*np.pi)**(dim/2)*np.linalg.det(cov)**0.5)
            val=np.append(val, multivariate_pdf)
        return np.array([val]).T
    
    
    #Apply a trained bayes classifier
    def useBayes(Xte,model):
        Pw1X = pdfGauss(Xte,model['mean1'],model['cov1'])*model['Pw1']
        Pw2X = pdfGauss(Xte,model['mean2'],model['cov2'])*model['Pw2']
        
        return ((-np.sign(Pw1X-Pw2X))*0.5+1.5).squeeze() #1-W1, 2-W2
        # Adicionado o sinal negativo para ter em conta a approach original do toolbox em matlab
        
        
    
    def plotDecision(X,y,model,granularity=0.01,classContours=False,f1name="f1",f2name="f2"):
        #Identify class indexes
        ix1=np.where(y==1)[0]
        ix2=np.where(y==2)[0]
        
        #Create vectors for plotting
        xRange=np.arange(X[:,0].min(),X[:,0].max()+0.5,granularity)
        yRange=np.arange(X[:,1].min(),X[:,1].max(),granularity)
        
        Xp, Yp = np.meshgrid(xRange, yRange)
        xy = np.column_stack([Xp.flat, Yp.flat])
        XX = np.array([Xp.ravel(), Yp.ravel()]).T
        #Sample class one pdf, just to get contours
        Z1 = pdfGauss(XX,model['mean1'],model['cov1'])
        Z1 = Z1.reshape(Xp.shape)
        #Sample class two pdf, just to get contours
        Z2=pdfGauss(XX,model['mean2'],model['cov2'])
        Z2 = Z2.reshape(Xp.shape)
        
        #Plot samples
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=X[ix1,0],y=X[ix1,1],name='1'))
        fig.add_trace(go.Scatter(x=X[ix2,0],y=X[ix2,1],name='2'))
        fig.update_traces(marker=dict(size=8))
        #Plot means
        fig.add_trace(go.Scatter(x=np.array(model['mean1'][0]),y=np.array(model['mean1'][1]), marker_size=20,
                            marker_symbol='x',name='1 mean',
                            marker=dict(color="blue",line=dict(width=1,
                                            color='black'))))
        fig.add_trace(go.Scatter(x=np.array(model['mean2'][0]),y=np.array(model['mean2'][1]), marker_size=20,
                            marker_symbol='x',name='2 mean',
                            marker=dict(color="red",line=dict(width=1,
                                            color='black'))))
        fig.update_traces(mode='markers')
        
        
        #Compute Bayes decision for the grid created by xRange and yRange
        yteP=np.zeros((yRange.shape[0],xRange.shape[0]))
        for i in range(xRange.shape[0]):
            for j in range(yRange.shape[0]):
                yteP[j,i]=useBayes(np.array([[xRange[i],yRange[j]]]),model)
        
        if classContours:
            #Add class 1 contours
            fig.add_trace(go.Contour(
                z=Z1,x=xRange,y=yRange,
                contours_coloring='lines',
                line_width=1,showscale=False,))
        
            #Add class 2 contours
            fig.add_trace(go.Contour(
                z=Z2,x=xRange,y=yRange,
                contours_coloring='lines',
                line_width=1,showscale=False,))
            
        
        #Add decision contour
        fig.add_trace(go.Contour(
            z=yteP,x=xRange,y=yRange,contours_coloring='lines',name='Decision Boundary',colorscale='Greys',
            line_width=4,showscale=False))
        fig.update_traces(ncontours=100, selector=dict(type='contour'))
        fig.update_xaxes(title_text=f1name)
        fig.update_yaxes(title_text=f2name)
        fig.update_layout(autosize=False,width=900,height=800)
        fig.show()
    
    
    
    
    
    """

    print(f"Healthy mean (mu_0): {mean1}")
    print(f"Cancer mean (mu_1): {mean2}")

    model = {
        "feature_names": feature_names,
        "mean0": mean1, # Healthy mean
        "mean1": mean2, # Cancer mean
        "cov0": cov1,   # Healthy covariance
        "cov1": cov2,   # Cancer covariance
        "Pw0": Pw0,     # Healthy prior
        "Pw1": Pw1      # Cancer prior
    }

    return model


def test_bayesian_gaussian(model, data=None , ixHealthy=None, ixCancer=None):
    """
    Classifies test data using the Gaussian Bayesian model and computes metrics.
    """
    if ixHealthy is None or ixCancer is None:
        ixHealthy = np.where(data["Classification"] == "Healthy")
        ixCancer = np.where(data["Classification"] == "Cancer")

    feature_names = model["feature_names"]
    
    # --- Extract Model Parameters ---
    mean0, mean1 = model["mean0"], model["mean1"]
    cov0, cov1 = model["cov0"], model["cov1"]
    Pw0, Pw1 = model["Pw0"], model["Pw1"]
    
    # --- Test Data Preparation ---
    X_healthy = data[feature_names].iloc[ixHealthy[0]].values
    X_cancer = data[feature_names].iloc[ixCancer[0]].values
    
    X_test = np.concatenate([X_healthy, X_cancer], axis=0)
    y_true = np.concatenate([np.zeros(len(X_healthy)), np.ones(len(X_cancer))])  # 0=Healthy, 1=Cancer

    # --- Prediction (using Log-Posteriors for numerical stability) ---
    # Log-likelihood + Log-prior = Log-posterior (log $P(x|w_i) + \log P(w_i)$)
    
    # P(x|w_0) * P(w_0) -> log posterior for Healthy (class 0)
    logp0 = multivariate_normal.logpdf(X_test, mean=mean0, cov=cov0) + np.log(Pw0)
    # P(x|w_1) * P(w_1) -> log posterior for Cancer (class 1)
    logp1 = multivariate_normal.logpdf(X_test, mean=mean1, cov=cov1) + np.log(Pw1)

    # Classification: $y = 1$ (Cancer) if $\log P(w_1|x) > \log P(w_0|x)$, i.e., $\log p_1 > \log p_0$.
    y_pred = np.where(logp1 >= logp0, 1, 0) # 1=Cancer, 0=Healthy
    
    # Decision scores for ROC (log posterior ratio)
    # Higher score favours class 1 (Cancer)
    decision_scores = logp1 - logp0 

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