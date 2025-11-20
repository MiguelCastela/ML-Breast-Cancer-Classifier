from xml.sax.handler import all_features
import ROC_AUC as roc_auc_module
import kruskal_wallis as kruskal_wallis_module
from PCA import pca_analysis, pca_scree, pca_kaiser 
from LDA import lda_analysis
from classifiers import run_all_classifiers, run_classifiers_multiple_times
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from data_loader import get_random_train_test_split








def one_run():
    """Main function that orchestrates all analyses"""

    dados, ixHealthy, ixCancer, fnames, Classes, train_data, test_data, ixHealthy_train, ixCancer_train, ixHealthy_test, ixCancer_test = get_random_train_test_split()
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
        lda = lda_analysis(train_data, ixHealthy_train, ixCancer_train)
        lda_test = lda_analysis(test_data, ixHealthy_test, ixCancer_test)
        print("LDA for each sample    : ", lda)
    except Exception as e:
        print(f"LDA analysis failed: {e}")
    
    print("\n6. CLASSIFIERS WITH DIFFERENT FEATURE SELECTIONS")
    print("-" * 50)
    try:

        
        classifier_results = run_all_classifiers(
            train_data=train_data,
            test_data=test_data,
            ixHealthy_train=ixHealthy_train,
            ixCancer_train=ixCancer_train,
            ixHealthy_test=ixHealthy_test,
            ixCancer_test=ixCancer_test,
            all_features=all_features,
            top5_roc=top5_roc,
            top5_kruskall=top5_kruskall,
            X_pca_train=X_pca,
            X_pca_test=X_pca_test,
            LD1_train=lda,
            LD1_test=lda_test
        )
        
        print("\n" + "="*60)
        print("CLASSIFIER RESULTS SUMMARY")
        print("="*60)

    
    except Exception as e:
        print(f"Classifier analysis failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("COMPLETED")
    print("=" * 60)

    return classifier_results, roc, H_results, fnames


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
    run_all_classifiers_multiple_times()