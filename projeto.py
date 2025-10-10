import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import plotly.express as px 
import scipy.stats as stats


#Read XLS      
xlsFile=pd.read_excel("dataR2.xlsx",usecols=["Age","BMI","Glucose","Insulin","HOMA","Leptin","Adiponectin","Resistin","MCP.1","Classification"])
dados=pd.DataFrame(data=xlsFile.dropna()) 

ixHealthy=np.where(dados['Classification']==1)
ixCancer=np.where(dados['Classification']==2)

#Define classes
Classes=["Healthy","Cancer"]

#Transform the numeric class labels in string labels
dados['Classification'].iloc[ixHealthy]=Classes[0]
dados['Classification'].iloc[ixCancer]=Classes[1]

fnames=dados.columns[0:-1]

#---------Kruskal-Wallis H-test---------

H_rank=[]

for feature in fnames:
    H,p= stats.kruskal(dados[feature].iloc[ixHealthy],
                       dados[feature].iloc[ixCancer])
    H_rank.append((feature,H))

print("Kruskal-Wallis H-test results (feature, H-statistic):")

Hs=sorted(H_rank, key=lambda x: x[1], reverse=True)

for f in Hs:
    print(f"Feature: {f[0]}, H-statistic: {f[1]:.4f}")

#---------ROC, AUC---------

ixHealthyCancer=np.concatenate((ixHealthy[0],ixCancer[0]))
y=dados['Classification'].to_numpy()[ixHealthyCancer]
roc_auc=np.zeros(fnames.shape)
i=0

for f in fnames:#Go along features

    fpr, tpr, _= roc_curve(y,dados[f].to_numpy()[ixHealthyCancer],pos_label="Cancer")

    figR = go.Figure()
    figR.add_scatter(x=fpr, y=tpr,mode='lines+markers')
    figR.update_layout(autosize=False,width=700,height=700,title=dict(text=f))
    figR.update_xaxes(title_text="1-SP",range=[-0.01, 1.01])
    figR.update_yaxes(title_text="SS",range=[-0.01, 1.01])
    

    roc_auc[i] = auc(fpr, tpr)#Compute area under the ROC curve
    
    figR.add_annotation(x=0.5, y=0.5,
            text="AUC: "+str(roc_auc[i]),
            showarrow=False,
            yshift=10)
    figR.show()
    i=i+1

sortIx=np.flip(np.argsort(roc_auc))#Sort using AUC
print("Sorting accourding to ROC-AUC:")
for i in sortIx:
    print(fnames[i]+"-->"+str(roc_auc[i]))

#Se AUC for a feature is 0.5, it means that the feature does not discriminate between the two classes
# ou seja, a feature aumenta com Cancer

#------CORRELATION MATRIX-------
X=np.array([dados[Hs[0][0]],dados[Hs[1][0]],dados[Hs[2][0]],dados[Hs[3][0]],dados[Hs[4][0]],dados[Hs[5][0]],dados[Hs[6][0]],dados[Hs[7][0]],dados[Hs[8][0]]])

corrMat=np.corrcoef(X)
fig= px.imshow(corrMat, x=[Hs[0][0],Hs[1][0],Hs[2][0],Hs[3][0],Hs[4][0],Hs[5][0],Hs[6][0],Hs[7][0],Hs[8][0]], y=[Hs[0][0],Hs[1][0],Hs[2][0],Hs[3][0],Hs[4][0],Hs[5][0],Hs[6][0],Hs[7][0],Hs[8][0]], color_continuous_scale='RdBu', title='Correlation Matrix of Top 9 Features')
fig.show()

#PCA

#---------PCA---------

X=dados.to_numpy()[:,:-1] #todas as colunas menos a última 
y=dados.to_numpy()[:,9] #só a coluna da Classificação
X=X.astype(float)
#Normalize
X=(X-np.mean(X,axis=0))/np.std(X,axis=0)

from sklearn.decomposition import PCA
pca = PCA()
pca.fit(X)
#PCA eigenvalues/Explained variance
print("PCA eigenvalues/Explained variance")
print(pca.explained_variance_)
print("Sum of eigenvalues="+str(np.sum(pca.explained_variance_)))
#PCA eigenvectors/Principal components
print("PCA eigenvectors/Principal components")
W=pca.components_.T
print(W)






