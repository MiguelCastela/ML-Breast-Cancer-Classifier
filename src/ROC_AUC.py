import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.metrics import roc_curve, auc
import plotly.express as px 



def roc_auc_analysis(dados, ixHealthy, ixCancer, fnames):
    ixHealthyCancer=np.concatenate((ixHealthy[0],ixCancer[0]))
    y=dados['Classification'].to_numpy()[ixHealthyCancer]
    roc_auc=np.zeros(fnames.shape)
    

    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[f for f in fnames],
        horizontal_spacing=0.08,
        vertical_spacing=0.08
    )
    i=0
    for r in range(3):
        for c in range(3):
            if i >= len(fnames):
                break

            f=fnames[i] 
            fpr, tpr, _= roc_curve(y,dados[f].to_numpy()[ixHealthyCancer],pos_label="Cancer")
            roc_auc[i] = auc(fpr, tpr) #Compute area under the ROC curve

            fig.add_trace(
                go.Scatter(x=fpr, y=tpr, mode='lines+markers', name=f"{f}"),
                row=r+1, col=c+1
            )

            fig.add_annotation(
                x=0.5, y=0.1,
                xref=f"x{(r*3 + c + 1)}", yref=f"y{(r*3 + c + 1)}",
                text=f"AUC: {roc_auc[i]:.3f}",
                showarrow=False,
                font=dict(size=10, color="black")
            )

            fig.update_xaxes(title_text="False Positive Rate", range=[-0.01, 1.01], row=r+1, col=c+1)
            fig.update_yaxes(title_text="True Positive Rate", range=[-0.01, 1.01], row=r+1, col=c+1)

            i += 1

        """ 
        figR = go.Figure()
        figR.add_scatter(x=fpr, y=tpr,mode='lines+markers')
        figR.update_layout(autosize=False,width=700,height=700,title=dict(text=f))
        figR.update_xaxes(title_text="1-SP",range=[-0.01, 1.01])
        figR.update_yaxes(title_text="SS",range=[-0.01, 1.01])
        

        
        figR.add_annotation(x=0.5, y=0.5,
                text="AUC: "+str(roc_auc[i]),
                showarrow=False,
                yshift=10)
        figR.show()
        """
    fig.update_layout(
        title_text="ROC Curves for All Features",
        autosize=False,
        width=1200,
        height=1200,
        showlegend=False
    )
    fig.show()
    # Adjust ROC-AUC scores to be >= 0.5
    roc_auc = np.maximum(roc_auc, 1 - roc_auc) 
    sortIx=np.flip(np.argsort(roc_auc))
    print("Sorting accourding to ROC-AUC:")
    for i in sortIx:
        print(fnames[i]+"-->"+str(roc_auc[i]))

    return roc_auc


def correlation_matrix(dados, Hs):
    X=np.array([dados[Hs[0][0]],dados[Hs[1][0]],dados[Hs[2][0]],dados[Hs[3][0]],dados[Hs[4][0]],dados[Hs[5][0]],dados[Hs[6][0]],dados[Hs[7][0]],dados[Hs[8][0]]])

    corrMat=np.corrcoef(X)
    fig= px.imshow(corrMat, x=[Hs[0][0],Hs[1][0],Hs[2][0],Hs[3][0],Hs[4][0],Hs[5][0],Hs[6][0],Hs[7][0],Hs[8][0]], y=[Hs[0][0],Hs[1][0],Hs[2][0],Hs[3][0],Hs[4][0],Hs[5][0],Hs[6][0],Hs[7][0],Hs[8][0]], color_continuous_scale='RdBu', title='Correlation Matrix of Top 9 Features')
    fig.show()
