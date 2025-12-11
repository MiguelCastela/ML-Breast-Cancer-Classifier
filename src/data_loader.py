import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



def get_random_train_test_split():



    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "data", "dataR2.xlsx")

    xlsFile=pd.read_excel(file_path,usecols=["Age","BMI","Glucose","Insulin","HOMA","Leptin","Adiponectin","Resistin","MCP.1","Classification"])
    dados=pd.DataFrame(data=xlsFile.dropna())

    ixHealthy=np.where(dados['Classification']==1)
    ixCancer=np.where(dados['Classification']==2)

    #Define classes
    Classes=["Healthy","Cancer"]

    #Transform the numeric class labels in string labels
    dados['Classification'] = dados['Classification'].astype(object)

    dados.loc[dados['Classification'] == 1, 'Classification'] = Classes[0]
    dados.loc[dados['Classification'] == 2, 'Classification'] = Classes[1]

    fnames=dados.columns[0:-1]




    X = dados[fnames]
    y = dados["Classification"]

    random_seed = 22

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,       # 30% left for test+validation
        stratify=y,           # maintain class proportions
        random_state=random_seed
    )

    
    mu = X_train.mean(axis=0)
    st = X_train.std(axis=0)

    # Normalize training data
    X_train = (X_train - mu) / st



    #scaler = StandardScaler()
    #X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=fnames, index=X_train.index)


    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,        # half of 30% for validation and half for testing
        stratify=y_temp,      # maintain class proportions again
        random_state=random_seed
    )


    X_val = (X_val - mu) / st
    X_test = (X_test - mu) / st


    #X_val   = pd.DataFrame(scaler.transform(X_val), columns=fnames, index=X_val.index)   
    #X_test = pd.DataFrame(scaler.transform(X_test), columns=fnames, index=X_test.index)

    # Merge X_train and y_train back into a single DataFrame
    train_data = X_train.copy()
    train_data['Classification'] = y_train.values

    # Build separate validation and test DataFrames (no concatenation)
    val_data = X_val.copy()
    val_data['Classification'] = y_val.values

    test_data = X_test.copy()
    test_data['Classification'] = y_test.values

    ixHealthy_train = np.where(train_data['Classification'] == "Healthy")
    ixCancer_train = np.where(train_data['Classification'] == "Cancer")

    ixHealthy_val = np.where(val_data['Classification'] == "Healthy")
    ixCancer_val = np.where(val_data['Classification'] == "Cancer")

    ixHealthy_test = np.where(test_data['Classification'] == "Healthy")
    ixCancer_test = np.where(test_data['Classification'] == "Cancer")

    print("Train shape:", train_data.shape)
    print("Validation shape:", val_data.shape)
    print("Test shape:", test_data.shape)

    print("\nClass distribution:")
    print("Train:\n", y_train.value_counts(normalize=True))
    print("Val:\n", y_val.value_counts(normalize=True))
    print("Test:\n", y_test.value_counts(normalize=True))

    return (
        dados,
        ixHealthy,
        ixCancer,
        fnames,
        Classes,
        train_data,
        val_data,
        test_data,
        ixHealthy_train,
        ixCancer_train,
        ixHealthy_val,
        ixCancer_val,
        ixHealthy_test,
        ixCancer_test,
    )

