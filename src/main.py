# Suppress specific sklearn FutureWarnings for AdaBoost 'algorithm' deprecation
import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="sklearn.ensemble._weight_boosting"
)

# Remove incorrect import of all_features
import ROC_AUC as roc_auc_module
import kruskal_wallis as kruskal_wallis_module
from PCA import pca_analysis, pca_scree, pca_kaiser 
from LDA import lda_analysis, lda_test
from classifiers import run_all_classifiers
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from data_loader import get_random_train_test_split
from run_grid_search import search_optimal_params_kfold
from collections import Counter


def one_run():
    """Main function that orchestrates all analyses"""

    dados, ixHealthy, ixCancer, fnames, Classes, train_data, val_data,   test_data, ixHealthy_train, ixCancer_train, ixHealthy_val, ixCancer_val, ixHealthy_test, ixCancer_test = get_random_train_test_split()
    all_features = list(fnames)

    print("=" * 60)
    print("META 1")
    print("=" * 60)
    
    print("\n1. DATA LOADING")
    print("-" * 30)
    print(f"Dataset shape: {dados.shape}")
    print(f"Features: {list(fnames)}")
    print(f"Classes: {Classes}")
    print(f"Healthy samples: {len(ixHealthy[0])}")
    print(f"Cancer samples: {len(ixCancer[0])}")
    



    print("\nPlotting interactive violin plots for 9 features (Healthy vs Cancer)...")

    Labels = ["Healthy", "Cancer"]
    ixs = [ixHealthy[0], ixCancer[0]]

    selected_features = list(fnames[:9])

    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=selected_features,
        shared_xaxes=False,
        shared_yaxes=False,
        horizontal_spacing=0.08,
        vertical_spacing=0.10
    )

    i = 0
    for r in range(3):
        for c in range(3):
            if i >= len(selected_features):
                break

            f = selected_features[i]

            for k, ix in enumerate(ixs):
                fig.add_trace(
                    go.Violin(
                        x=[k] * len(ix),          
                        y=dados[f].iloc[ix],
                        name=Labels[k],
                        legendgroup=Labels[k],
                        scalegroup=f,
                        box_visible=True,
                        meanline_visible=True,
                        points='all',
                        width=0.6,
                        fillcolor='rgba(0,0,255,0.3)' if Labels[k] == 'Healthy' else 'rgba(255,0,0,0.3)',
                        line_color='blue' if Labels[k] == 'Healthy' else 'red',
                        showlegend=(i == 0)
                    ),
                    row=r + 1, col=c + 1
                )

            fig.update_xaxes(
                tickmode='array',
                tickvals=[0, 1],
                ticktext=Labels,
                range=[-0.5, 1.5],
                tickfont=dict(size=12),
                row=r + 1, col=c + 1
            )

            fig.update_yaxes(
                title_text=f,
                tickfont=dict(size=12),
                row=r + 1, col=c + 1
            )

            i += 1

    fig.update_layout(
        title_text="",
        height=950,
        width=1150,
        violinmode='group',
        template='plotly_white',
        font=dict(size=14)
    )

    fig.write_html("plots/violin_plots_9_features.html")


    
    print("\n2. ROC-AUC ANALYSIS")
    print("-" * 30)
    roc = roc_auc_module.roc_auc_analysis(train_data, ixHealthy_train, ixCancer_train, fnames)
    top5_roc = [fnames[i] for i in roc.argsort()[-5:][::-1]]

    print("Top 5 features from ROC-AUC Analysis:", top5_roc)
    top5_roc_scores = [roc[i] for i in roc.argsort()[-5:][::-1]]
    print("Top 5 ROC-AUC scores:", top5_roc_scores)


    
    
    print("\n3. KRUSKAL-WALLIS H-TEST")
    print("-" * 30)
    H_results = kruskal_wallis_module.kruskal_wallis_test(train_data, ixHealthy_train, ixCancer_train, fnames)
    top5_kruskall = [f[0] for f in H_results[:5]]
    print("Top 5 features from Kruskal-Wallis Test:", top5_kruskall)
    top5_kruskall_stats = [f[1] for f in H_results[:5]]
    print("Top 5 Kruskal-Wallis H-statistics:", top5_kruskall_stats)

    print("\n3.1 CORRELATION MATRIX OF TOP 9 FEATURES")
    print("-" * 40)
    try:
        if H_results and len(H_results) > 0:
            top_features = H_results[:9] if len(H_results) >= 9 else H_results
            roc_auc_module.correlation_matrix(train_data, top_features)
        else:
            print("No results from Kruskal-Wallis test to create correlation matrix")
    except Exception as e:
        print(f"Could not create correlation matrix: {e}")
    
    print("\n4. PRINCIPAL COMPONENT ANALYSIS (PCA)")
    print("-" * 40)
    try:
        evr, numpcs, pca_model, X, y = pca_analysis(train_data)
        top_pca = evr.argsort()[-numpcs:][::-1]
        top_pcs = [f"PC{i+1}" for i in top_pca]
        print("Top" + str(numpcs) + " principal components from PCA Analysis:", top_pcs , evr)
        X_pca = pca_model.transform(X)[:, :numpcs]  # shape: (n_samples, numpcs)


        X_pca_val = pca_model.transform(val_data.to_numpy()[:, :-1])[:, :numpcs]
        X_pca_test = pca_model.transform(test_data.to_numpy()[:, :-1])[:, :numpcs]  





        print("\n4.1 PCA SCREE PLOT")
        print("-" * 20)
        num_kaiser, kaiser_indices, kaiser_evr = pca_kaiser(pca_model, X, y)


        print("\nTop PCs by Kaiser Criterion:")
        for i, val in zip(kaiser_indices, kaiser_evr):
            print(f"  PC{i+1}: {val*100:.2f}% of variance")

        print("\n4.2 KAISER CRITERION PCA")
        print("-" * 25)
        pca_scree(pca_model)

    except Exception as e:
        print(f"PCA analysis failed: {e}")
    
    print("\n5. LINEAR DISCRIMINANT ANALYSIS (LDA)")
    print("-" * 40)
    try:
        lda, lda_model = lda_analysis(train_data)
        lda_test_values = lda_test(test_data, lda_model)
        lda_val = lda_test(val_data, lda_model)
        print("LDA for each sample    : ", lda)
    except Exception as e:
        print(f"LDA analysis failed: {e}")
    
    print("\n6. CLASSIFIERS WITH DIFFERENT FEATURE SELECTIONS")
    print("-" * 50)
    
    print("\n6. HYPERPARAMETER SEARCH (Validation Step)")
    print("-" * 50)

    
    # 6a. K-FOLD cross-validated search (train-only)
    best_params_kfold, metrics_summary_kfold = search_optimal_params_kfold(
        train_data=train_data,              # pandas DataFrame with training rows only
        all_features=all_features,          # list of original feature column names
        top5_roc=top5_roc,                  # list of top-5 ROC columns
        top5_kruskall=top5_kruskall,         # list of top-5 KW columns
        k=5,                                 # number of folds
        X_pca_all=X_pca,                   # numpy array aligned to train_data rows
        LD1_all=lda,                       # numpy array (N x 1) aligned to train_data rows
        progress_every=10,                  # print progress more frequently
        enable_custom_svm=False,
    )

    print("\nBest hyperparameters (k-fold):")
    for key, val in best_params_kfold.items():
        print(f"  {key}: {val}")
    print("\nMetrics summary (avg over folds):")
    for feat_set, methods in metrics_summary_kfold.items():
        print(f"[{feat_set}]")
        for method, mets in methods.items():
            print(f"  {method}: avg F1={mets['avg_f1']*100:.2f}% | avg Acc={mets['avg_acc']*100:.2f}%")
    """
    # 6b. Hold-out search using explicit validation split
    print("\nRunning hold-out hyperparameter search (train vs val)...")
    best_params_holdout = search_optimal_params(
        train_data=train_data,
        val_data=val_data,
        ixHealthy_train=ixHealthy_train,
        ixCancer_train=ixCancer_train,
        ixHealthy_val=ixHealthy_val,
        ixCancer_val=ixCancer_val,
        all_features=all_features,
        top5_roc=top5_roc,
        top5_kruskall=top5_kruskall,
        X_pca_train=X_pca,
        X_pca_val=X_pca_val,
        LD1_train=lda,
        LD1_val=lda_val,
        enable_custom_svm=False,
        early_stop_patience=6,
        early_stop_min_delta=0.05,
        early_stop_warmup_linear=12,
        early_stop_warmup_rbf_c=4,
        early_stop_warmup_rbf_gamma=12
        
    )

    print("\nBest hyperparameters (hold-out):")
    for key, val in best_params_holdout.items():
        print(f"  {key}: {val}")
    print("\n--- HYPERPARAMETER SEARCH COMPLETE ---")
    """

    # Return best params along with analysis results so wrappers can use one_run()
    # best_params_kfold may be undefined if k-fold block is commented; return None for consistency
    return roc, H_results, fnames, None, best_params_kfold
    # try:

        
    #     classifier_results = run_all_classifiers(
    #         train_data=train_data,
    #         test_data=test_data,
    #         ixHealthy_train=ixHealthy_train,
    #         ixCancer_train=ixCancer_train,
    #         ixHealthy_test=ixHealthy_test,
    #         ixCancer_test=ixCancer_test,
    #         all_features=all_features,
    #         top5_roc=top5_roc,
    #         top5_kruskall=top5_kruskall,
    #         X_pca_train=X_pca,
    #         X_pca_test=X_pca_test,
    #         LD1_train=lda,
    #         LD1_test=lda_test_values
    #     )
        
    #     print("\n" + "="*60)
    #     print("CLASSIFIER RESULTS SUMMARY")
    #     print("="*60)

    
    # except Exception as e:
    #     print(f"Classifier analysis failed: {e}")
    #     import traceback
    #     traceback.print_exc()

    print("\n" + "=" * 60)
    print("COMPLETED")
    print("=" * 60)

    return roc, H_results, fnames


def run_all_classifiers_multiple_times():
    all_results = {}
    roc_results = []
    H_results_all = []
    fnames = []
    n_runs = 5
    for run in range(n_runs):
        print(f"\n--- Run {run + 1} ---")

        classifier_results, roc, H_results, fnames = one_run()

        roc_results.append({fnames[i]: val for i, val in enumerate(roc)})
        H_results_all.append({f[0]: f[1] for f in H_results})

        all_features = fnames 

  
    

        for key, metrics in classifier_results.items():
            if key not in all_results:
                all_results[key] = []   
            all_results[key].append(metrics)

    # Compute mean metrics
    mean_results = {k: tuple(np.mean(v, axis=0)) for k, v in all_results.items()}

    # Print mean results
    print("\n=== Average metrics over", n_runs, "runs ===")
    for clf, metrics in mean_results.items():
        print(f"{clf}: Accuracy={metrics[0]:.3f}, Sensitivity={metrics[1]:.3f}, "
            f"Specificity={metrics[2]:.3f}, Precision={metrics[3]:.3f}, "
            f"F1={metrics[4]:.3f}, ROC-AUC={metrics[5]:.3f}")
    
    
    median_roc_values = {f: np.median([run[f] for run in roc_results]) for f in all_features}
    median_H_values = {f: np.median([run[f] for run in H_results_all]) for f in all_features}

    print("\nMedian ROC-AUC values across runs (sorted):")
    for f, val in sorted(median_roc_values.items(), key=lambda x: x[1], reverse=True):
        print(f"{f} --> {val:.4f}")

    # Sorted by descending H-stat
    print("\nMedian Kruskal-Wallis H-statistic values across runs (sorted):")
    for f, val in sorted(median_H_values.items(), key=lambda x: x[1], reverse=True):
        print(f"{f} --> {val:.4f}")



if __name__ == "__main__":
    one_run()
    #run_all_classifiers_multiple_times()
    #grid_search_multiple_times()