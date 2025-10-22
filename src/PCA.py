from sklearn.decomposition import PCA
import numpy as np
import plotly.express as px 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd

# Global variables to store PCA results for use across functions
global_pca = None
global_X = None
global_y = None

#---------PCA---------
def pca_analysis(dados):
    global global_pca, global_X, global_y
    
    X=dados.to_numpy()[:,:-1] #todas as colunas menos a última 
    y=dados.to_numpy()[:,-1] #só a coluna da Classificação

    X=X.astype(float)
    
    #Normalize
    mu = np.mean(X, axis=0)
    st = np.std(X, axis=0)
    X=(X-mu)/st

    pca_model = PCA()
    pca_model.fit(X)
    
    # Store for use in other functions
    global_pca = pca_model
    global_X = X
    global_y = y

    print("PCA eigenvalues/Explained variance")
    print(pca_model.explained_variance_)
    print("Sum of eigenvalues="+str(np.sum(pca_model.explained_variance_)))
    print("cumulative explained variance ratio" + str(np.cumsum(pca_model.explained_variance_ratio_)))
    print("PCA eigenvectors/Principal components")
    W=pca_model.components_.T
    print(W)

    Xp = X @ W[:, 0]  # projection of all samples onto PC1

    fig=px.scatter(x=Xp,y=np.zeros(np.shape(X)[0]),color=y,labels=dict(x="PC1",y="",color="Class"), title="PCA - Projection onto PC1")
    fig.update_traces(marker_size=5)
    fig.show()

    
    # Scree plot data
    evr = pca_model.explained_variance_ratio_
    df = pd.DataFrame({
        "PC": np.arange(1, len(evr)+1),
        "Explained Variance Ratio": evr
    })
    
    # Scree plot
    fig = px.bar(df, x="PC", y="Explained Variance Ratio",
                 text=df["Explained Variance Ratio"].round(2),
                 title="PCA Scree Plot")
    fig.update_traces(textposition='outside')
    fig.update_layout(yaxis=dict(title="Explained Variance Ratio"), xaxis=dict(title="Principal Component"))
    fig.show()
    

    
    
    return pca_model, X, y

def pca_kaiser(dados):
    global global_X, global_y
    
    # If pca_analysis hasn't been run yet, run it first
    if global_X is None or global_y is None:
        pca_analysis(dados)
    
    fig = px.scatter(x=np.arange(1,len(global_pca.explained_variance_)+1,1), y=global_pca.explained_variance_,
                    labels=dict(x="PC",y="Explained Variance"))
    fig.add_hline(y=1,line_width=3, line_dash="dash", line_color="red")
    fig.update_traces(marker_size=10)
    fig.show()


    num_components = np.sum(global_pca.explained_variance_ > 1)
    print(f"Number of eigenvalues > 1 (Kaiser criterion): {num_components}")

    kaiser_indices = np.where(global_pca.explained_variance_ > 1)[0] + 1  # +1 to make them 1-indexed (PC1, PC2, ...)
    print("Components with eigenvalues > 1:", kaiser_indices)

    variance_retained = (np.sum(global_pca.explained_variance_[kaiser_indices - 1] ** 2) / np.sum(global_pca.explained_variance_ ** 2)) * 100
    print(f"Variance retained according to Kaiser: {variance_retained:.2f}%")

    #plot projected data according kaiser criterion
    pca_kaiser = PCA(n_components=1)
    X1D = pca_kaiser.fit_transform(global_X)
    print(np.shape(X1D))

    fig=px.scatter(x=X1D[:,0],y=np.zeros(np.shape(X1D)[0]),color=global_y,labels=dict(x="PC1",y="",color="Class"), title ="Kaiser Criterion PCA - Projection onto PC1")
    fig.update_traces(marker_size=8)
    fig.update_xaxes(title_text="PC1")
    fig.show()




def pca_scree(dados):
    global global_pca
    
    # If pca_analysis hasn't been run yet, run it first
    if global_pca is None:
        pca_analysis(dados)
    
    #Scree plot


