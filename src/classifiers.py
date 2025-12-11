import numpy as np
import pandas as pd
import numpy as np
from classifier_euclidean import train_euclidean_distance, test_euclidean_distance
from classifier_mahalanobis import train_mahalanobis_distance, test_mahalanobis_distance
from classifier_fisher import train_fisher_lda, test_fisher_lda
from classifier_svm import train_svm_linear, test_svm_generic
# Removed invalid imports of non-existent grid_search helpers
from classifier_knn import train_knn, test_knn
from classifier_decision_tree import train_decision_tree, test_decision_tree
from classifier_bayes import train_bayesian_classifier, test_bayesian_gaussian
from classifier_adaboost import train_adaboost, test_adaboost
from classifier_adaboost import train_adaboost_custom, test_adaboost_custom
from classifier_svm import train_svm_custom, train_svm_rbf_kernel






def run_all_classifiers(train_data, test_data, ixHealthy_train, ixCancer_train, ixHealthy_test, ixCancer_test, all_features, top5_roc=None, top5_kruskall=None, X_pca_train=None, X_pca_test=None, LD1_train=None, LD1_test=None):
   

    """
    Run classifiers using different feature sets and train/test split:
    - train_data / test_data: pandas DataFrames with the same columns
    - top5_roc / top5_kruskall: lists of column names from original dataframe
    - X_pca_train / X_pca_test: numpy arrays for PCA features
    - LD1_train / LD1_test: numpy arrays for LDA1 (1D) features
    """
    results = {}




    model = train_euclidean_distance(all_features, method_name="All Features", 
                                    data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
    results['all_euclidean'] = test_euclidean_distance(model, data=test_data, 
                                                    ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

    model = train_mahalanobis_distance(all_features, method_name="All Features", 
                                    data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
    results['all_mahalanobis'] = test_mahalanobis_distance(model, data=test_data, 
                                                        ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

    model = train_fisher_lda(all_features, method_name="All Features", 
                            data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
    results['all_fisher'] = test_fisher_lda(model, data=test_data, 
                                            ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

    # SVM (Linear)
    model = train_svm_linear(all_features, method_name="All Features", 
                             data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
    results['all_svm'] = test_svm_generic(model, data=test_data, 
                                  ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

    # SVM (Custom Kernel)

    model = train_svm_custom(all_features, method_name="All Features", 
                             data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
    
    results['all_svm_custom'] = test_svm_generic(model, data=test_data, 
                                               ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
    
    # SVM (RBF Kernel)

    model = train_svm_rbf_kernel(all_features, method_name="All Features",
                                    data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
    
    results['all_svm_rbf'] = test_svm_generic(model, data=test_data,
                                        ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
    
    # KNN
    model = train_knn(all_features, method_name="All Features", 
                      data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
    results['all_knn'] = test_knn(model, data=test_data, 
                                  ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

    # Decision Tree
    model = train_decision_tree(all_features, method_name="All Features", 
                                data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
    results['all_decision_tree'] = test_decision_tree(model, data=test_data, 
                                                      ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

    # Bayesian Gaussian
    model = train_bayesian_classifier(all_features, method_name="All Features", 
                                      data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
    results['all_bayes'] = test_bayesian_gaussian(model, data=test_data, 
                                                  ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

    # AdaBoost
    model = train_adaboost(all_features, method_name="All Features", 
                           data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
    results['all_adaboost'] = test_adaboost(model, data=test_data, 
                                            ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

    # Adaboost Custom
    model = train_adaboost_custom(all_features, method_name="All Features", 
                           data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
    
    results['all_adaboost_custom'] = test_adaboost_custom(model, data=test_data, 
                                            ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
    
    if top5_roc is not None:
        model = train_euclidean_distance(top5_roc, method_name="ROC-AUC", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['roc_euclidean'] = test_euclidean_distance(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_mahalanobis_distance(top5_roc, method_name="ROC-AUC", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['roc_mahalanobis'] = test_mahalanobis_distance(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_fisher_lda(top5_roc, method_name="ROC-AUC", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['roc_fisher'] = test_fisher_lda(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_svm_linear(top5_roc, method_name="ROC-AUC", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['roc_svm'] = test_svm_generic(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_svm_custom(top5_roc, method_name="ROC-AUC", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['roc_svm_custom'] = test_svm_generic(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_svm_rbf_kernel(top5_roc, method_name="ROC-AUC", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['roc_svm_rbf'] = test_svm_generic(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_knn(top5_roc, method_name="ROC-AUC", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['roc_knn'] = test_knn(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_decision_tree(top5_roc, method_name="ROC-AUC", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['roc_decision_tree'] = test_decision_tree(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_bayesian_classifier(top5_roc, method_name="ROC-AUC", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['roc_bayes'] = test_bayesian_gaussian(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_adaboost(top5_roc, method_name="ROC-AUC", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['roc_adaboost'] = test_adaboost(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_adaboost_custom(top5_roc, method_name="ROC-AUC", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['roc_adaboost_custom'] = test_adaboost_custom(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)  
                                                              
    if top5_kruskall is not None:
        model = train_euclidean_distance(top5_kruskall, method_name="Kruskal-Wallis", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['kw_euclidean'] = test_euclidean_distance(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_mahalanobis_distance(top5_kruskall, method_name="Kruskal-Wallis", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['kw_mahalanobis'] = test_mahalanobis_distance(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_fisher_lda(top5_kruskall, method_name="Kruskal-Wallis", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['kw_fisher'] = test_fisher_lda(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_svm_linear(top5_kruskall, method_name="Kruskal-Wallis", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['kw_svm'] = test_svm_generic(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_svm_custom(top5_kruskall, method_name="Kruskal-Wallis", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['kw_svm_custom'] = test_svm_generic(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_svm_rbf_kernel(top5_kruskall, method_name="Kruskal-Wallis", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['kw_svm_rbf'] = test_svm_generic(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_knn(top5_kruskall, method_name="Kruskal-Wallis", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['kw_knn'] = test_knn(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_decision_tree(top5_kruskall, method_name="Kruskal-Wallis", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['kw_decision_tree'] = test_decision_tree(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_bayesian_classifier(top5_kruskall, method_name="Kruskal-Wallis", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['kw_bayes'] = test_bayesian_gaussian(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_adaboost(top5_kruskall, method_name="Kruskal-Wallis", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['kw_adaboost'] = test_adaboost(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_adaboost_custom(top5_kruskall, method_name="Kruskal-Wallis", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['kw_adaboost_custom'] = test_adaboost_custom(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)


    if X_pca_train is not None and X_pca_test is not None:
        df_pca_train = pd.DataFrame(X_pca_train, columns=[f"PC{i+1}" for i in range(X_pca_train.shape[1])])
        df_pca_test = pd.DataFrame(X_pca_test, columns=[f"PC{i+1}" for i in range(X_pca_test.shape[1])])
        pca_features = list(df_pca_train.columns)

        model = train_euclidean_distance(pca_features, method_name="PCA", data=df_pca_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['pca_euclidean'] = test_euclidean_distance(model, data=df_pca_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_mahalanobis_distance(pca_features, method_name="PCA", data=df_pca_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['pca_mahalanobis'] = test_mahalanobis_distance(model, data=df_pca_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_fisher_lda(pca_features, method_name="PCA", data=df_pca_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['pca_fisher'] = test_fisher_lda(model, data=df_pca_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_svm_linear(pca_features, method_name="PCA", data=df_pca_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['pca_svm'] = test_svm_generic(model, data=df_pca_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_svm_custom(pca_features, method_name="PCA", data=df_pca_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['pca_svm_custom'] = test_svm_generic(model, data=df_pca_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_svm_rbf_kernel(pca_features, method_name="PCA", data=df_pca_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['pca_svm_rbf'] = test_svm_generic(model, data=df_pca_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_knn(pca_features, method_name="PCA", data=df_pca_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['pca_knn'] = test_knn(model, data=df_pca_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_decision_tree(pca_features, method_name="PCA", data=df_pca_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['pca_decision_tree'] = test_decision_tree(model, data=df_pca_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_bayesian_classifier(pca_features, method_name="PCA", data=df_pca_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['pca_bayes'] = test_bayesian_gaussian(model, data=df_pca_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_adaboost(pca_features, method_name="PCA", data=df_pca_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['pca_adaboost'] = test_adaboost(model, data=df_pca_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_adaboost_custom(pca_features, method_name="PCA", data=df_pca_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['pca_adaboost_custom'] = test_adaboost_custom(model, data=df_pca_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
    if LD1_train is not None and LD1_test is not None:


        df_lda_train = pd.DataFrame(LD1_train, columns=['LD1'])
        df_lda_test = pd.DataFrame(LD1_test, columns=['LD1'])


        model = train_fisher_lda(['LD1'], method_name="LDA", data=df_lda_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['lda_fisher'] = test_fisher_lda(model, data=df_lda_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
    

        model = train_euclidean_distance(['LD1'], method_name="LDA", data=df_lda_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['lda_euclidean'] = test_euclidean_distance(model, data=df_lda_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_mahalanobis_distance(['LD1'], method_name="LDA", data=df_lda_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['lda_mahalanobis'] = test_mahalanobis_distance(model, data=df_lda_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_svm_linear(['LD1'], method_name="LDA", data=df_lda_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['lda_svm'] = test_svm_generic(model, data=df_lda_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_svm_custom(['LD1'], method_name="LDA", data=df_lda_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['lda_svm_custom'] = test_svm_generic(model, data=df_lda_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_svm_rbf_kernel(['LD1'], method_name="LDA", data=df_lda_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['lda_svm_rbf'] = test_svm_generic(model, data=df_lda_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_knn(['LD1'], method_name="LDA", data=df_lda_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['lda_knn'] = test_knn(model, data=df_lda_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_decision_tree(['LD1'], method_name="LDA", data=df_lda_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['lda_decision_tree'] = test_decision_tree(model, data=df_lda_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_bayesian_classifier(['LD1'], method_name="LDA", data=df_lda_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['lda_bayes'] = test_bayesian_gaussian(model, data=df_lda_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_adaboost(['LD1'], method_name="LDA", data=df_lda_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['lda_adaboost'] = test_adaboost(model, data=df_lda_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_adaboost_custom(['LD1'], method_name="LDA", data=df_lda_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['lda_adaboost_custom'] = test_adaboost_custom(model, data=df_lda_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
    return results

