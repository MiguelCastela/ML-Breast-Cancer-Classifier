import numpy as np
import pandas as pd
import numpy as np
from src.classifier_euclidean import train_euclidean_distance, test_euclidean_distance
from src.classifier_mahalanobis import train_mahalanobis_distance, test_mahalanobis_distance
from src.classifier_fisher import train_fisher_lda, test_fisher_lda







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


    
    if top5_roc is not None:
        model = train_euclidean_distance(top5_roc, method_name="ROC-AUC", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['roc_euclidean'] = test_euclidean_distance(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_mahalanobis_distance(top5_roc, method_name="ROC-AUC", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['roc_mahalanobis'] = test_mahalanobis_distance(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_fisher_lda(top5_roc, method_name="ROC-AUC", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['roc_fisher'] = test_fisher_lda(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

    if top5_kruskall is not None:
        model = train_euclidean_distance(top5_kruskall, method_name="Kruskal-Wallis", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['kw_euclidean'] = test_euclidean_distance(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_mahalanobis_distance(top5_kruskall, method_name="Kruskal-Wallis", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['kw_mahalanobis'] = test_mahalanobis_distance(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_fisher_lda(top5_kruskall, method_name="Kruskal-Wallis", data=train_data, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['kw_fisher'] = test_fisher_lda(model, data=test_data, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

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

    if LD1_train is not None and LD1_test is not None:


        df_lda_train = pd.DataFrame(LD1_train, columns=['LD1'])
        df_lda_test = pd.DataFrame(LD1_test, columns=['LD1'])


        model = train_fisher_lda(['LD1'], method_name="LDA", data=df_lda_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['lda_fisher'] = test_fisher_lda(model, data=df_lda_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)
    

        model = train_euclidean_distance(['LD1'], method_name="LDA", data=df_lda_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['lda_euclidean'] = test_euclidean_distance(model, data=df_lda_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

        model = train_mahalanobis_distance(['LD1'], method_name="LDA", data=df_lda_train, ixHealthy=ixHealthy_train, ixCancer=ixCancer_train)
        results['lda_mahalanobis'] = test_mahalanobis_distance(model, data=df_lda_test, ixHealthy=ixHealthy_test, ixCancer=ixCancer_test)

    return results


def run_classifiers_multiple_times(train_data, test_data, ixHealthy_train, ixCancer_train, ixHealthy_test, ixCancer_test,
                                   all_features, top5_roc=None, top5_kruskall=None, X_pca_train=None, X_pca_test=None,
                                   LD1_train=None, LD1_test=None, n_runs=5):
    """
    Run run_all_classifiers n_runs times and compute mean metrics.
    """
    all_results = {}

    for run in range(n_runs):
        print(f"\n--- Run {run + 1} ---")
        results = run_all_classifiers(
            train_data=train_data,
            test_data=test_data,
            ixHealthy_train=ixHealthy_train,
            ixCancer_train=ixCancer_train,
            ixHealthy_test=ixHealthy_test,
            ixCancer_test=ixCancer_test,
            all_features=all_features,
            top5_roc=top5_roc,
            top5_kruskall=top5_kruskall,
            X_pca_train=X_pca_train,
            X_pca_test=X_pca_test,
            LD1_train=LD1_train,
            LD1_test=LD1_test
        )

        for key, metrics in results.items():
            if key not in all_results:
                all_results[key] = []
            all_results[key].append(metrics)

    # Compute mean metrics
    mean_results = {k: tuple(np.mean(v, axis=0)) for k, v in all_results.items()}

    # Print mean results
    print("\n=== Average metrics over", n_runs, "runs ===")
    for clf, metrics in mean_results.items():
        print(f"{clf}: Accuracy={metrics[0]:.3f}, Sensitivity={metrics[1]:.3f}, Specificity={metrics[2]:.3f}, "
              f"Precision={metrics[3]:.3f}, F1={metrics[4]:.3f}, ROC-AUC={metrics[5]:.3f}")

    return mean_results
    


          
                                 

