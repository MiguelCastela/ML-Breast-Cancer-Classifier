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
from classifier_random_forest import train_random_forest, test_random_forest






def run_all_classifiers(train_data, test_data, ixHealthy_train, ixCancer_train, ixHealthy_test, ixCancer_test, all_features, top5_roc=None, top5_kruskall=None, X_pca_train=None, X_pca_test=None, LD1_train=None, LD1_test=None):
   

    """
    Run classifiers using different feature sets and train/test split:
    - train_data / test_data: pandas DataFrames with the same columns
    - top5_roc / top5_kruskall: lists of column names from original dataframe
    - X_pca_train / X_pca_test: numpy arrays for PCA features
    - LD1_train / LD1_test: numpy arrays for LDA1 (1D) features
    """
    results = {}
    train_results = {}




    model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_euclidean_distance(all_features, method_name="All Features", 
                                    data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
    results['all_euclidean'] = test_euclidean_distance(model, data=test_data, 
                                                    ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
    train_results['all_euclidean'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)

    model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_mahalanobis_distance(all_features, method_name="All Features", 
                                    data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
    results['all_mahalanobis'] = test_mahalanobis_distance(model, data=test_data, 
                                                        ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
    train_results['all_mahalanobis'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)

    model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_fisher_lda(all_features, method_name="All Features", 
                            data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
    results['all_fisher'] = test_fisher_lda(model, data=test_data, 
                                            ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
    train_results['all_fisher'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)

    # SVM (Linear)
    model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_svm_linear(all_features, method_name="All Features", 
                             data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train, C=2.0)
    results['all_svm'] = test_svm_generic(model, data=test_data, 
                                  ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
    train_results['all_svm'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)

    # SVM (Custom Kernel)

    model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_svm_custom(all_features, method_name="All Features", 
                             data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train, C=2.0)
    
    results['all_svm_custom'] = test_svm_generic(model, data=test_data, 
                                               ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
    train_results['all_svm_custom'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)
    
    # SVM (RBF Kernel)

    model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_svm_rbf_kernel(all_features, method_name="All Features",
                                    data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train,
                                    C=8.0, gamma=0.0625)
    
    results['all_svm_rbf'] = test_svm_generic(model, data=test_data,
                                        ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
    train_results['all_svm_rbf'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)
    
    # KNN
    model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_knn(all_features, method_name="All Features", 
                      data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train,
                      n_neighbors=17)
    results['all_knn'] = test_knn(model, data=test_data, 
                                  ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
    train_results['all_knn'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)

    # Decision Tree
    model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_decision_tree(all_features, method_name="All Features", 
                                data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train,
                                max_depth=7, criterion='gini')
    results['all_decision_tree'] = test_decision_tree(model, data=test_data, 
                                                      ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
    train_results['all_decision_tree'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)

    # Bayesian Gaussian
    model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_bayesian_classifier(all_features, method_name="All Features", 
                                      data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
    results['all_bayes'] = test_bayesian_gaussian(model, data=test_data, 
                                                  ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
    train_results['all_bayes'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)

    # AdaBoost
    model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_adaboost(all_features, method_name="All Features", 
                           data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train,
                           n_estimators=25, learning_rate=0.2)
    results['all_adaboost'] = test_adaboost(model, data=test_data, 
                                            ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
    train_results['all_adaboost'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)

    # Adaboost Custom
    model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_adaboost_custom(all_features, method_name="All Features", 
                           data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
    
    results['all_adaboost_custom'] = test_adaboost_custom(model, data=test_data, 
                                            ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
    train_results['all_adaboost_custom'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)

    # Random Forest
    model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_random_forest(
        all_features, method_name="All Features",
        data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train,
        n_estimators=100, max_depth=7
    )
    results['all_random_forest'] = test_random_forest(model, data=test_data,
                                                      ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
    train_results['all_random_forest'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)
    
    if top5_roc is not None:
        model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_euclidean_distance(top5_roc, method_name="ROC-AUC", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['roc_euclidean'] = test_euclidean_distance(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
        train_results['roc_euclidean'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)

        model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_mahalanobis_distance(top5_roc, method_name="ROC-AUC", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['roc_mahalanobis'] = test_mahalanobis_distance(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
        train_results['roc_mahalanobis'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)

        model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_fisher_lda(top5_roc, method_name="ROC-AUC", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['roc_fisher'] = test_fisher_lda(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
        train_results['roc_fisher'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)

        model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_svm_linear(top5_roc, method_name="ROC-AUC", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train, C=1.0)
        results['roc_svm'] = test_svm_generic(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
        train_results['roc_svm'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)

        model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_svm_custom(top5_roc, method_name="ROC-AUC", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train, C=1.0)
        results['roc_svm_custom'] = test_svm_generic(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
        train_results['roc_svm_custom'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)

        model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_svm_rbf_kernel(top5_roc, method_name="ROC-AUC", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train,
                         C=0.5, gamma=0.0625)
        results['roc_svm_rbf'] = test_svm_generic(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
        train_results['roc_svm_rbf'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)

        model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_knn(top5_roc, method_name="ROC-AUC", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train, n_neighbors=5)
        results['roc_knn'] = test_knn(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
        train_results['roc_knn'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)

        model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_decision_tree(top5_roc, method_name="ROC-AUC", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train,
                        max_depth=2, criterion='gini')
        results['roc_decision_tree'] = test_decision_tree(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
        train_results['roc_decision_tree'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)

        model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_bayesian_classifier(top5_roc, method_name="ROC-AUC", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['roc_bayes'] = test_bayesian_gaussian(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
        train_results['roc_bayes'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)

        model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_adaboost(top5_roc, method_name="ROC-AUC", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train,
                       n_estimators=75, learning_rate=0.2)
        results['roc_adaboost'] = test_adaboost(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
        train_results['roc_adaboost'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)

        model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_adaboost_custom(top5_roc, method_name="ROC-AUC", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['roc_adaboost_custom'] = test_adaboost_custom(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)  
        train_results['roc_adaboost_custom'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)
        # Random Forest
        model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_random_forest(
            top5_roc, method_name="ROC-AUC", data=train_data,
            ixHealthy=ixHealthy_train, ixCancer=ixCancer_train,
            n_estimators=15, max_depth=10
        )
        results['roc_random_forest'] = test_random_forest(model, data=test_data,
                                                          ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
        train_results['roc_random_forest'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)
                                                              
    if top5_kruskall is not None:
        model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_euclidean_distance(top5_kruskall, method_name="Kruskal-Wallis", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['kw_euclidean'] = test_euclidean_distance(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
        train_results['kw_euclidean'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)

        model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_mahalanobis_distance(top5_kruskall, method_name="Kruskal-Wallis", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['kw_mahalanobis'] = test_mahalanobis_distance(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
        train_results['kw_mahalanobis'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)

        model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_fisher_lda(top5_kruskall, method_name="Kruskal-Wallis", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['kw_fisher'] = test_fisher_lda(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
        train_results['kw_fisher'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)

        model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_svm_linear(top5_kruskall, method_name="Kruskal-Wallis", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train, C=0.03125)
        results['kw_svm'] = test_svm_generic(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
        train_results['kw_svm'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)

        model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_svm_custom(top5_kruskall, method_name="Kruskal-Wallis", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train, C=0.03125)
        results['kw_svm_custom'] = test_svm_generic(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
        train_results['kw_svm_custom'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)

        model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_svm_rbf_kernel(top5_kruskall, method_name="Kruskal-Wallis", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train,
                         C=0.5, gamma=1.0)
        results['kw_svm_rbf'] = test_svm_generic(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
        train_results['kw_svm_rbf'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)

        model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_knn(top5_kruskall, method_name="Kruskal-Wallis", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train, n_neighbors=9)
        results['kw_knn'] = test_knn(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
        train_results['kw_knn'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)

        model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_decision_tree(top5_kruskall, method_name="Kruskal-Wallis", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train,
                        max_depth=2, criterion='gini')
        results['kw_decision_tree'] = test_decision_tree(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
        train_results['kw_decision_tree'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)

        model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_bayesian_classifier(top5_kruskall, method_name="Kruskal-Wallis", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['kw_bayes'] = test_bayesian_gaussian(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
        train_results['kw_bayes'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)

        model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_adaboost(top5_kruskall, method_name="Kruskal-Wallis", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train,
                       n_estimators=75, learning_rate=0.2)
        results['kw_adaboost'] = test_adaboost(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
        train_results['kw_adaboost'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)
        model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_adaboost_custom(top5_kruskall, method_name="Kruskal-Wallis", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['kw_adaboost_custom'] = test_adaboost_custom(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
        train_results['kw_adaboost_custom'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)
        # Random Forest
        model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_random_forest(
            top5_kruskall, method_name="Kruskal-Wallis", data=train_data,
            ixHealthy=ixHealthy_train, ixCancer=ixCancer_train,
            n_estimators=50, max_depth=10
        )
        results['kw_random_forest'] = test_random_forest(model, data=test_data,
                                                         ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
        train_results['kw_random_forest'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)


    if X_pca_train is not None and X_pca_test is not None:
        df_pca_train = pd.DataFrame(X_pca_train, columns=[f"PC{i+1}" for i in range(X_pca_train.shape[1])])
        df_pca_test = pd.DataFrame(X_pca_test, columns=[f"PC{i+1}" for i in range(X_pca_test.shape[1])])
        pca_features = list(df_pca_train.columns)

        model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_euclidean_distance(pca_features, method_name="PCA", data=df_pca_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['pca_euclidean'] = test_euclidean_distance(model, data=df_pca_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
        train_results['pca_euclidean'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)

        model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_mahalanobis_distance(pca_features, method_name="PCA", data=df_pca_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['pca_mahalanobis'] = test_mahalanobis_distance(model, data=df_pca_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
        train_results['pca_mahalanobis'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)

        model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_fisher_lda(pca_features, method_name="PCA", data=df_pca_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['pca_fisher'] = test_fisher_lda(model, data=df_pca_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
        train_results['pca_fisher'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)

        model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_svm_linear(pca_features, method_name="PCA", data=df_pca_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train, C=0.125)
        results['pca_svm'] = test_svm_generic(model, data=df_pca_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
        train_results['pca_svm'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)

        model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_svm_custom(pca_features, method_name="PCA", data=df_pca_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train, C=0.125)
        results['pca_svm_custom'] = test_svm_generic(model, data=df_pca_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
        train_results['pca_svm_custom'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)

        model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_svm_rbf_kernel(pca_features, method_name="PCA", data=df_pca_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train,
                         C=0.5, gamma=0.5)
        results['pca_svm_rbf'] = test_svm_generic(model, data=df_pca_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
        train_results['pca_svm_rbf'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)

        model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_knn(pca_features, method_name="PCA", data=df_pca_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train, n_neighbors=17)
        results['pca_knn'] = test_knn(model, data=df_pca_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
        train_results['pca_knn'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)

        model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_decision_tree(pca_features, method_name="PCA", data=df_pca_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train,
                        max_depth=7, criterion='gini')
        results['pca_decision_tree'] = test_decision_tree(model, data=df_pca_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
        train_results['pca_decision_tree'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)

        model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_bayesian_classifier(pca_features, method_name="PCA", data=df_pca_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['pca_bayes'] = test_bayesian_gaussian(model, data=df_pca_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
        train_results['pca_bayes'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)

        model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_adaboost(pca_features, method_name="PCA", data=df_pca_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train,
                       n_estimators=75, learning_rate=0.2)
        results['pca_adaboost'] = test_adaboost(model, data=df_pca_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
        train_results['pca_adaboost'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)

        model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_adaboost_custom(pca_features, method_name="PCA", data=df_pca_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['pca_adaboost_custom'] = test_adaboost_custom(model, data=df_pca_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
        train_results['pca_adaboost_custom'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)
        # Random Forest
        model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_random_forest(
            pca_features, method_name="PCA", data=df_pca_train,
            ixHealthy=ixHealthy_train, ixCancer=ixCancer_train,
            n_estimators=25, max_depth=2
        )
        results['pca_random_forest'] = test_random_forest(model, data=df_pca_test,
                                                          ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
        train_results['pca_random_forest'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)
    if LD1_train is not None and LD1_test is not None:


        df_lda_train = pd.DataFrame(LD1_train, columns=['LD1'])
        df_lda_test = pd.DataFrame(LD1_test, columns=['LD1'])


        model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_fisher_lda(['LD1'], method_name="LDA", data=df_lda_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['lda_fisher'] = test_fisher_lda(model, data=df_lda_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
        train_results['lda_fisher'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)
    

        model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_euclidean_distance(['LD1'], method_name="LDA", data=df_lda_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['lda_euclidean'] = test_euclidean_distance(model, data=df_lda_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
        train_results['lda_euclidean'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)

        model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_mahalanobis_distance(['LD1'], method_name="LDA", data=df_lda_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['lda_mahalanobis'] = test_mahalanobis_distance(model, data=df_lda_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
        train_results['lda_mahalanobis'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)

        model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_svm_linear(['LD1'], method_name="LDA", data=df_lda_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train, C=0.03125)
        results['lda_svm'] = test_svm_generic(model, data=df_lda_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
        train_results['lda_svm'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)

        model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_svm_custom(['LD1'], method_name="LDA", data=df_lda_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train, C=0.03125)
        results['lda_svm_custom'] = test_svm_generic(model, data=df_lda_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
        train_results['lda_svm_custom'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)

        model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_svm_rbf_kernel(['LD1'], method_name="LDA", data=df_lda_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train,
                         C=0.125, gamma=0.5)
        results['lda_svm_rbf'] = test_svm_generic(model, data=df_lda_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
        train_results['lda_svm_rbf'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)

        model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_knn(['LD1'], method_name="LDA", data=df_lda_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train,
                  n_neighbors=1)
        results['lda_knn'] = test_knn(model, data=df_lda_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
        train_results['lda_knn'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)

        model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_decision_tree(['LD1'], method_name="LDA", data=df_lda_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train,
                        max_depth=2, criterion='gini')
        results['lda_decision_tree'] = test_decision_tree(model, data=df_lda_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
        train_results['lda_decision_tree'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)

        model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_bayesian_classifier(['LD1'], method_name="LDA", data=df_lda_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['lda_bayes'] = test_bayesian_gaussian(model, data=df_lda_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
        train_results['lda_bayes'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)

        model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_adaboost(['LD1'], method_name="LDA", data=df_lda_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train,
                       n_estimators=10, learning_rate=0.05)
        results['lda_adaboost'] = test_adaboost(model, data=df_lda_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
        train_results['lda_adaboost'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)

        model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_adaboost_custom(['LD1'], method_name="LDA", data=df_lda_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['lda_adaboost_custom'] = test_adaboost_custom(model, data=df_lda_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
        train_results['lda_adaboost_custom'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)
        # Random Forest
        model, tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc = train_random_forest(
            ['LD1'], method_name="LDA", data=df_lda_train,
            ixHealthy=ixHealthy_train, ixCancer=ixCancer_train,
            n_estimators=15, max_depth=1
        )
        results['lda_random_forest'] = test_random_forest(model, data=df_lda_test,
                                                          ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
        train_results['lda_random_forest'] = (tr_acc, tr_sens, tr_spec, tr_prec, tr_f1, tr_auc)
    return results, train_results

