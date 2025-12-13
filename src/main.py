# Suppress specific sklearn FutureWarnings for AdaBoost 'algorithm' deprecation
import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="sklearn.ensemble._weight_boosting"
)

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
import random
from collections import defaultdict, Counter


def one_run():

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

    try:


        classifier_results, train_results = run_all_classifiers(
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
            LD1_test=lda_test_values
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

    return classifier_results, train_results, roc, H_results, fnames


def run_all_classifiers_multiple_times():
    all_test_results = {}
    all_train_results = {}
    roc_results = []
    H_results_all = []
    fnames = []
    n_runs = 5
    for run in range(n_runs):
        print(f"\n--- Run {run + 1} ---")

        classifier_results, train_results, roc, H_results, fnames = one_run()

        roc_results.append({fnames[i]: val for i, val in enumerate(roc)})
        H_results_all.append({f[0]: f[1] for f in H_results})

        all_features = fnames 

  
    

        for key, metrics in classifier_results.items():
            if key not in all_test_results:
                all_test_results[key] = []
            all_test_results[key].append(metrics)

        for key, metrics in train_results.items():
            if key not in all_train_results:
                all_train_results[key] = []
            all_train_results[key].append(metrics)

    mean_test_results = {k: tuple(np.mean(v, axis=0)) for k, v in all_test_results.items()}
    std_test_results = {k: tuple(np.std(v, axis=0, ddof=0)) for k, v in all_test_results.items()}
    mean_train_results = {k: tuple(np.mean(v, axis=0)) for k, v in all_train_results.items()}
    std_train_results = {k: tuple(np.std(v, axis=0, ddof=0)) for k, v in all_train_results.items()}

    print("\n=== TEST metrics over", n_runs, "runs (mean ± std) ===")
    for clf in mean_test_results.keys():
        m = mean_test_results[clf]
        s = std_test_results[clf]
        print(
            f"{clf}: "
            f"Accuracy={m[0]:.3f}±{s[0]:.3f}, Sensitivity={m[1]:.3f}±{s[1]:.3f}, "
            f"Specificity={m[2]:.3f}±{s[2]:.3f}, Precision={m[3]:.3f}±{s[3]:.3f}, "
            f"F1={m[4]:.3f}±{s[4]:.3f}, ROC-AUC={m[5]:.3f}±{s[5]:.3f}"
        )

    print("\n=== TRAIN metrics over", n_runs, "runs (mean ± std) ===")
    for clf in mean_train_results.keys():
        m = mean_train_results[clf]
        s = std_train_results[clf]
        print(
            f"{clf}: "
            f"Accuracy={m[0]:.3f}±{s[0]:.3f}, Sensitivity={m[1]:.3f}±{s[1]:.3f}, "
            f"Specificity={m[2]:.3f}±{s[2]:.3f}, Precision={m[3]:.3f}±{s[3]:.3f}, "
            f"F1={m[4]:.3f}±{s[4]:.3f}, ROC-AUC={m[5]:.3f}±{s[5]:.3f}"
        )
    
    
    median_roc_values = {f: np.median([run[f] for run in roc_results]) for f in all_features}
    median_H_values = {f: np.median([run[f] for run in H_results_all]) for f in all_features}

    print("\nMedian ROC-AUC values across runs (sorted):")
    for f, val in sorted(median_roc_values.items(), key=lambda x: x[1], reverse=True):
        print(f"{f} --> {val:.4f}")

    # Sorted by descending H-stat
    print("\nMedian Kruskal-Wallis H-statistic values across runs (sorted):")
    for f, val in sorted(median_H_values.items(), key=lambda x: x[1], reverse=True):
        print(f"{f} --> {val:.4f}")

def run_kfold_hyperparam_multiple_times(n_runs=5):

    agg_metrics = defaultdict(
        lambda: defaultdict(
            lambda: {
                "train_f1": [], "train_acc": [],
                "val_f1": [], "val_acc": []
            }
        )
    )
    param_votes = defaultdict(lambda: Counter())

    for run in range(n_runs):
        # Use one_run to generate a fresh split/seed and compute k-fold search
        _roc, _H_results, _fnames, best_params_kfold, metrics_summary_kfold = one_run()

        # Aggregate train and validation metrics across runs
        for feat_set, methods in metrics_summary_kfold.items():
            for method, m in methods.items():
                agg_metrics[feat_set][method]["train_f1"].append(m.get("train_avg_f1", np.nan))
                agg_metrics[feat_set][method]["train_acc"].append(m.get("train_avg_acc", np.nan))
                agg_metrics[feat_set][method]["val_f1"].append(m.get("val_avg_f1", np.nan))
                agg_metrics[feat_set][method]["val_acc"].append(m.get("val_avg_acc", np.nan))

        # Count votes for parameters (majority vote per key)
        for key, val in best_params_kfold.items():
            param_votes[key][str(val)] += 1

    # Compute aggregates: mean and std over runs
    metrics_summary_over_runs = {}
    for feat_set, methods in agg_metrics.items():
        metrics_summary_over_runs[feat_set] = {}
        for method, vals in methods.items():
            tr_f1_arr = np.array(vals["train_f1"], dtype=float)
            tr_acc_arr = np.array(vals["train_acc"], dtype=float)
            f1_arr = np.array(vals["val_f1"], dtype=float)
            acc_arr = np.array(vals["val_acc"], dtype=float)
            metrics_summary_over_runs[feat_set][method] = {
                "train_avg_f1": float(np.nanmean(tr_f1_arr)),
                "train_std_f1": float(np.nanstd(tr_f1_arr)),
                "train_avg_acc": float(np.nanmean(tr_acc_arr)),
                "train_std_acc": float(np.nanstd(tr_acc_arr)),
                "val_avg_f1": float(np.nanmean(f1_arr)),
                "val_std_f1": float(np.nanstd(f1_arr)),
                "val_avg_acc": float(np.nanmean(acc_arr)),
                "val_std_acc": float(np.nanstd(acc_arr)),
            }

    # Decide winners via majority vote; tie → random
    final_params = {}
    winners_summary = {}
    for key, counter in param_votes.items():
        if not counter:
            final_params[key] = None
            winners_summary[key] = "no votes"
            continue
        most_common = counter.most_common()
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            # tie: random among tied values
            top_count = most_common[0][1]
            tied_vals = [val for val, cnt in most_common if cnt == top_count]
            chosen = random.choice(tied_vals)
            final_params[key] = chosen
            winners_summary[key] = f"random (tie) among {tied_vals} count={top_count}"
        else:
            chosen, cnt = most_common[0]
            final_params[key] = chosen
            winners_summary[key] = f"winner '{chosen}' with {cnt}/{n_runs} votes"

    # Print concise summary
    print("\n=== Aggregated Train/Validation Metrics over runs ===")
    for feat_set, methods in metrics_summary_over_runs.items():
        print(f"[{feat_set}]")
        for method, m in methods.items():
            print(
                f"  {method}: "
                f"train F1={m['train_avg_f1']*100:.2f}%±{m['train_std_f1']*100:.2f}% | "
                f"train Acc={m['train_avg_acc']*100:.2f}%±{m['train_std_acc']*100:.2f}% | "
                f"val F1={m['val_avg_f1']*100:.2f}%±{m['val_std_f1']*100:.2f}% | "
                f"val Acc={m['val_avg_acc']*100:.2f}%±{m['val_std_acc']*100:.2f}%"
            )

    print("\n=== Majority-Vote Best Hyperparameters ===")
    for key, val in final_params.items():
        print(f"  {key}: {val} -> {winners_summary[key]}")

    return metrics_summary_over_runs, final_params, winners_summary

if __name__ == "__main__":
    #one_run()
    run_all_classifiers_multiple_times()
    #run_kfold_hyperparam_multiple_times(n_runs=5)
