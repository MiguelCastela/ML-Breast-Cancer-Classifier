import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # <--- add this


#Get path to the data file
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "data", "dataR2.xlsx")

#Read XLS
xlsFile=pd.read_excel(file_path,usecols=["Age","BMI","Glucose","Insulin","HOMA","Leptin","Adiponectin","Resistin","MCP.1","Classification"])
dados=pd.DataFrame(data=xlsFile.dropna())

ixHealthy=np.where(dados['Classification']==1)
ixCancer=np.where(dados['Classification']==2)

#Define classes
Classes=["Healthy","Cancer"]

#Transform the numeric class labels in string labels
dados['Classification'].iloc[ixHealthy]=Classes[0]
dados['Classification'].iloc[ixCancer]=Classes[1]

fnames=dados.columns[0:-1]

scaler = StandardScaler()
dados[fnames] = scaler.fit_transform(dados[fnames])

# Separate features (X) and labels (y)
X = dados[fnames]
y = dados["Classification"]

# 1️⃣ First split: 70% training, 30% temporary (test + validation)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,       # 30% left for test+validation
    stratify=y,           # maintain class proportions
    random_state=5
)

# 2️⃣ Second split: split the temporary set into 15% test, 15% validation
# (i.e., split 30% into 50/50 halves)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,        # half of 30% = 15%
    stratify=y_temp,      # maintain class proportions again
    random_state=5
)

# Merge X_train and y_train back into a single DataFrame
train_data = X_train.copy()
train_data['Classification'] = y_train.values

# Combine validation and test sets for evaluation
test_data = pd.concat([X_val, X_test], axis=0)
test_labels = pd.concat([y_val, y_test], axis=0)
test_data['Classification'] = test_labels.values

ixHealthy_train = np.where(train_data['Classification'] == "Healthy")
ixCancer_train = np.where(train_data['Classification'] == "Cancer")

ixHealthy_test = np.where(test_data['Classification'] == "Healthy")
ixCancer_test = np.where(test_data['Classification'] == "Cancer")

# ✅ Check the resulting shapes
print("Train shape:", train_data.shape)
print("Validation shape:", X_val.shape)
print("Test shape:", X_test.shape)

# ✅ Optional: verify class balance
print("\nClass distribution:")
print("Train:\n", y_train.value_counts(normalize=True))
print("Val:\n", y_val.value_counts(normalize=True))
print("Test:\n", y_test.value_counts(normalize=True))
