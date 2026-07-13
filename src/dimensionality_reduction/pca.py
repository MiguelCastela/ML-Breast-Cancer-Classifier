from sklearn.decomposition import PCA
import numpy as np
import plotly.express as px 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd

def pca_analysis(dados, variance_threshold=0.95):
    
    X=dados.to_numpy()[:,:-1]
    y=dados.to_numpy()[:,-1] 
    X=X.astype(float)
    
    pca_model = PCA()
    pca_model.fit(X)
    

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
    fig.write_html("plots/pca_scatter_plot.html")

    
    evr = pca_model.explained_variance_ratio_
    df = pd.DataFrame({
        "PC": np.arange(1, len(evr)+1),
        "Explained Variance Ratio": evr
    })
    
    fig = px.bar(df, x="PC", y="Explained Variance Ratio",
                 text=df["Explained Variance Ratio"].round(2),
                 title="PCA Explained Variance Ratio per Principal Component")
    fig.update_traces(textposition='outside', textfont_size=16)
    fig.update_layout(yaxis=dict(title="Explained Variance Ratio", title_font_size=16, tickfont_size=14), xaxis=dict(title="Principal Component", title_font_size=16, tickfont_size=14),
    margin=dict(t=40, b=20, l=30, r=40),
    autosize=True,)
    #fig.show()
    fig.write_html("plots/explained_variance.html")
    
    
    return evr, n_components_95, pca_model, X, y

def pca_kaiser(pca_model, x, y):

    fig = px.scatter(x=np.arange(1,len(pca_model.explained_variance_)+1,1), y=pca_model.explained_variance_,
                    labels=dict(x="PC",y="Explained Variance"))
    fig.add_hline(y=1,line_width=3, line_dash="dash", line_color="red")
    fig.update_traces(marker_size=10)

    fig.update_layout(
        xaxis=dict(
            title=dict(text="Principal Component", font=dict(size=18)),  
            tickfont=dict(size=14)  
        ),
        yaxis=dict(
            title=dict(text="Explained Variance", font=dict(size=18)),
            tickfont=dict(size=14)
        ),
        margin=dict(l=40, r=20, t=30, b=40),  
        autosize=True
    )
    #fig.show()
    fig.write_html("plots/pca_kaiser_scree_plot.html")

    eigenvalues = pca_model.explained_variance_
    evr = pca_model.explained_variance_ratio_




    kaiser_indices = np.where(eigenvalues > 1)[0]  
    print("Components with eigenvalues > 1:", kaiser_indices + 1)


    num_components = len(kaiser_indices)
    print(f"Number of eigenvalues > 1 (Kaiser criterion): {num_components}")

    variance_retained = (np.sum(eigenvalues[kaiser_indices] ** 2) / np.sum(eigenvalues ** 2)) * 100

    print(f"Variance retained according to Kaiser (manual): {variance_retained:.2f}%")


    #plot projected data according kaiser criterion 
    pca_kaiser = PCA(n_components=1)
    X1D = pca_kaiser.fit_transform(x)
    print(np.shape(X1D))

    fig=px.scatter(x=X1D[:,0],y=np.zeros(np.shape(X1D)[0]),color=y,labels=dict(x="PC1",y="",color="Class"), title ="Kaiser Criterion PCA - Projection onto PC1")
    fig.update_traces(marker_size=8)
    fig.update_xaxes(title_text="PC1")
    #fig.show()
    fig.write_html("plots/kaiser_pca_scatter_plot.html")

    return num_components, kaiser_indices, evr[kaiser_indices]




def pca_scree(pca_model):


    print("Accourding to Scree we can subjectively say that the eighenvalue plot is never constant to decide on a number of components to retain.")
    


