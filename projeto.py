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

fnames=dados.columns[1:]#Get the feature names

H_rank=[]

for feature in fnames:
    H,p= stats.kruskal(dados[feature].iloc[ixHealthy],
                       dados[feature].iloc[ixCancer])
    H_rank.append((feature,H))

print("Kruskal-Wallis H-test results (feature, H-statistic):")

Hs=sorted(H_rank, key=lambda x: x[1], reverse=True)

for f in Hs:
    print(f"Feature: {f[0]}, H-statistic: {f[1]:.4f}")
