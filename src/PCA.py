from sklearn.decomposition import PCA
import numpy as np
import plotly.express as px 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd

global_pca = None
global_X = None
global_y = None

def pca_analysis(dados, variance_threshold=0.95):
    global global_pca, global_X, global_y
    
    X=dados.to_numpy()[:,:-1]
    y=dados.to_numpy()[:,-1] 
    X=X.astype(float)
    
    pca_model = PCA()
    pca_model.fit(X)
    
    global_pca = pca_model
    global_X = X
    global_y = y

    print("PCA eigenvalues/Explained variance")
    print(pca_model.explained_variance_)
    print("Sum of eigenvalues="+str(np.sum(pca_model.explained_variance_)))
    cumulative_variance = np.cumsum(pca_model.explained_variance_ratio_)
    print("cumulative explained variance ratio" + str(cumulative_variance))

    n_components_95 = np.argmax(cumulative_variance >= variance_threshold) + 1
    print(f"Number of components needed to explain {variance_threshold*100}% variance: {n_components_95}")
    print(f"Actual variance explained by {n_components_95} components: {cumulative_variance[n_components_95-1]*100:.2f}%")

    

    print("PCA eigenvectors/Principal components")
    W=pca_model.components_.T
    print(W)

    Xp = X @ W[:, 0]  # projection of all samples onto PC1

    fig=px.scatter(x=Xp,y=np.zeros(np.shape(X)[0]),color=y,labels=dict(x="PC1",y="",color="Class"), title="PCA - Projection onto PC1")
    fig.update_traces(marker_size=5)
    fig.show()

    
    evr = pca_model.explained_variance_ratio_
    df = pd.DataFrame({
        "PC": np.arange(1, len(evr)+1),
        "Explained Variance Ratio": evr
    })
    
    fig = px.bar(df, x="PC", y="Explained Variance Ratio",
                 text=df["Explained Variance Ratio"].round(2),
                 title="PCA Scree Plot")
    fig.update_traces(textposition='outside')
    fig.update_layout(yaxis=dict(title="Explained Variance Ratio"), xaxis=dict(title="Principal Component"))
    fig.show()
    

    
    
    return evr, n_components_95, pca_model, X, y

def pca_kaiser(dados):
    global global_X, global_y
    
    if global_X is None or global_y is None:
        pca_analysis(dados)
    
    fig = px.scatter(x=np.arange(1,len(global_pca.explained_variance_)+1,1), y=global_pca.explained_variance_,
                    labels=dict(x="PC",y="Explained Variance"))
    fig.add_hline(y=1,line_width=3, line_dash="dash", line_color="red")
    fig.update_traces(marker_size=10)
    fig.show()

    eigenvalues = global_pca.explained_variance_
    evr = global_pca.explained_variance_ratio_




    kaiser_indices = np.where(global_pca.explained_variance_ > 1)[0]  
    print("Components with eigenvalues > 1:", kaiser_indices + 1)


    num_components = len(kaiser_indices)
    print(f"Number of eigenvalues > 1 (Kaiser criterion): {num_components}")

    variance_retained = (np.sum(global_pca.explained_variance_[kaiser_indices] ** 2) / np.sum(global_pca.explained_variance_ ** 2)) * 100
    

    """
    variance_retained = (np.sum(global_pca.explained_variance_[kaiser_indices]) /
                     np.sum(global_pca.explained_variance_)) * 100
                     """
    

    print(f"Variance retained according to Kaiser (manual): {variance_retained:.2f}%")


    

    """
    variance_retained = np.sum(evr[kaiser_indices]) * 100
    

    print(f"\nTotal variance retained (Kaiser): {variance_retained:.2f}%")
    """

    #plot projected data according kaiser criterion 
    pca_kaiser = PCA(n_components=1)
    X1D = pca_kaiser.fit_transform(global_X)
    print(np.shape(X1D))

    fig=px.scatter(x=X1D[:,0],y=np.zeros(np.shape(X1D)[0]),color=global_y,labels=dict(x="PC1",y="",color="Class"), title ="Kaiser Criterion PCA - Projection onto PC1")
    fig.update_traces(marker_size=8)
    fig.update_xaxes(title_text="PC1")
    fig.show()

    return num_components, kaiser_indices, evr[kaiser_indices]




def pca_scree(dados):
    global global_pca
    
    if global_pca is None:
        pca_analysis(dados)

    print("Accourding to Scree we can subjectively say that the eighenvalue plot is never constant to decide on a number of components to retain.")
    


