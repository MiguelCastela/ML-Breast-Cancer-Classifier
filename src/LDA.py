import numpy as np
import plotly.express as px 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def lda_analysis(dados, ixHealthy, ixCancer):
    #---------LDA---------



    X=dados.to_numpy()[:,:-1] #todas as colunas menos a última 
    y=dados.to_numpy()[:,-1] #só a coluna da Classificação


    # prepare transform on dataset
    lda = LinearDiscriminantAnalysis()
    lda.fit(X,y)
    # apply transform to dataset
    transformed = lda.transform(X)  

    LD1 = transformed[:, 0]  # First linear discriminant

    print("LDA scalings (eigenvectors)")
    print(lda.scalings_)
    print("LDA explained variance ratio")
    print(lda.explained_variance_ratio_)


    #Plot transformed data
    fig=px.scatter(x=transformed[:,0],y=np.zeros(np.shape(X)[0]),color=y,labels=dict(x="LDA1" ,color="Class"), title ="")
    fig.update_traces(marker_size=10)
    fig.update_layout(
        xaxis=dict(
            title=dict(text="LDA1", font=dict(size=18)),  # X-axis title font
            tickfont=dict(size=14)                        # X-axis tick font
        ),
        yaxis=dict(
            title=dict(text=""),  # No title for y-axis (since it's just 0s)
            tickfont=dict(size=14),
            showticklabels=False  # Hide y-axis tick labels completely
        ),
        legend=dict(
            font=dict(size=14),
            title=dict(font=dict(size=16))
        ),
        margin=dict(l=40, r=20, t=40, b=40),  # tighten whitespace
        autosize=True
    )
    # fig.show()
    fig.write_html("plots/lda_projection_onto_lda1.html")
    

    return LD1

