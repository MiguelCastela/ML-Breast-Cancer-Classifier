from data_loader import dados, ixHealthy, ixCancer, fnames, Classes, train_data, test_data, ixHealthy_train, ixCancer_train, ixHealthy_test, ixCancer_test
import ROC_AUC as roc_auc_module
import kruskal_wallis as kruskal_wallis_module
from PCA import pca_analysis, pca_scree, pca_kaiser 
from LDA import lda_analysis
from classifiers import run_all_classifiers





def main():
    """Main function that orchestrates all analyses"""

    
    all_features = list(fnames)

    print("=" * 60)
    print("META 1")
    print("=" * 60)
    
    # Step 1: Data Loading (already done by importing data_loader)
    print("\n1. DATA LOADING")
    print("-" * 30)
    print(f"Dataset shape: {dados.shape}")
    print(f"Features: {list(fnames)}")
    print(f"Classes: {Classes}")
    print(f"Healthy samples: {len(ixHealthy[0])}")
    print(f"Cancer samples: {len(ixCancer[0])}")
    
    # Step 2: ROC-AUC Analysis
    print("\n2. ROC-AUC ANALYSIS")
    print("-" * 30)
    roc = roc_auc_module.roc_auc_analysis(train_data, ixHealthy_train, ixCancer_train, fnames)
    top5_roc = [fnames[i] for i in roc.argsort()[-5:][::-1]]
    print("Top 5 features from ROC-AUC Analysis:", top5_roc)
    top5_roc_scores = [roc[i] for i in roc.argsort()[-5:][::-1]]
    print("Top 5 ROC-AUC scores:", top5_roc_scores)


    
    
    # Step 3: Kruskal-Wallis Test
    print("\n3. KRUSKAL-WALLIS H-TEST")
    print("-" * 30)
    H_results = kruskal_wallis_module.kruskal_wallis_test(train_data, ixHealthy_train, ixCancer_train, fnames)
    top5_kruskall = [f[0] for f in H_results[:5]]
    print("Top 5 features from Kruskal-Wallis Test:", top5_kruskall)
    top5_kruskall_stats = [f[1] for f in H_results[:5]]
    print("Top 5 Kruskal-Wallis H-statistics:", top5_kruskall_stats)

    # Create correlation matrix with top features from Kruskal-Wallis
    print("\n3.1 CORRELATION MATRIX OF TOP 9 FEATURES")
    print("-" * 40)
    try:
        # Get top 9 features from Kruskal-Wallis results
        if H_results and len(H_results) > 0:
            top_features = H_results[:9] if len(H_results) >= 9 else H_results
            roc_auc_module.correlation_matrix(dados, top_features)
        else:
            print("No results from Kruskal-Wallis test to create correlation matrix")
    except Exception as e:
        print(f"Could not create correlation matrix: {e}")
    
    # Step 4: PCA Analysis
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
        num_kaiser, kaiser_indices, kaiser_evr = pca_kaiser(train_data)


        print("\nTop PCs by Kaiser Criterion:")
        for i, val in zip(kaiser_indices, kaiser_evr):
            print(f"  PC{i+1}: {val*100:.2f}% of variance")

        print("\n4.2 KAISER CRITERION PCA")
        print("-" * 25)
        pca_scree(train_data)

    except Exception as e:
        print(f"PCA analysis failed: {e}")
    
    # Step 5: LDA Analysis
    print("\n5. LINEAR DISCRIMINANT ANALYSIS (LDA)")
    print("-" * 40)
    try:
        lda = lda_analysis(train_data, ixHealthy_train, ixCancer_train)
        lda_test = lda_analysis(test_data, ixHealthy_test, ixCancer_test)
        print("LDA for each sample    : ", lda)
    except Exception as e:
        print(f"LDA analysis failed: {e}")
    
    # Step 6: Run Classifiers with different feature selection methods
    print("\n6. CLASSIFIERS WITH DIFFERENT FEATURE SELECTIONS")
    print("-" * 50)
    try:

        
        # Run all classifiers with the different feature selection results
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

if __name__ == "__main__":
    main()