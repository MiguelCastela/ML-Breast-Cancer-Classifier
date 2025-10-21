import numpy as np
import plotly.express as px 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def lda_analysis(dados, ixHealthy, ixCancer):
    #---------LDA---------



    X=dados.to_numpy()[:,:-1] #todas as colunas menos a última 
    y=dados.to_numpy()[:,-1] #só a coluna da Classificação

    X=X.astype(float)

    #Normalize
    X=(X-np.mean(X,axis=0))/np.std(X,axis=0)

    # prepare transform on dataset
    lda = LinearDiscriminantAnalysis()
    lda.fit(X,y)
    # apply transform to dataset
    transformed = lda.transform(X)

    #Plot transformed data
    fig=px.scatter(x=transformed[:,0],y=np.zeros(np.shape(X)[0]),color=y,labels=dict(x="LDA1" ,color="Class"), title ="LDA - Projection onto LDA1")
    fig.update_traces(marker_size=10)
    fig.show()


    """
    plot_df = pd.DataFrame({
    'LD1': transformed[:, 0],  # LDA first component
    'Class': y
    })

    fig = px.strip(
        plot_df,
        x='LD1',
        color='Class',
        title="LDA Projection onto First Component"
    )

    fig.update_traces(marker=dict(size=10, opacity=0.7))
    fig.update_layout(yaxis_title="")
    fig.show()

    """

