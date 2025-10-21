from data_loader import dados, ixHealthy, ixCancer, fnames, Classes
import ROC_AUC as roc_auc_module
import kruskal_wallis as kruskal_wallis_module
from PCA import pca_analysis, pca_scree, pca_kaiser 
from LDA import lda_analysis





def main():
    """Main function that orchestrates all analyses"""
    
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
    roc_auc_module.roc_auc_analysis(dados, ixHealthy, ixCancer, fnames)
    
    # Step 3: Kruskal-Wallis Test
    print("\n3. KRUSKAL-WALLIS H-TEST")
    print("-" * 30)
    H_results = kruskal_wallis_module.kruskal_wallis_test(dados, ixHealthy, ixCancer, fnames)
    
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
        pca_analysis(dados)
        print("\n4.1 PCA SCREE PLOT")
        print("-" * 20)
        pca_kaiser(dados)
        print("\n4.2 KAISER CRITERION PCA")
        print("-" * 25)
        pca_scree(dados)

    except Exception as e:
        print(f"PCA analysis failed: {e}")
    
    # Step 5: LDA Analysis
    print("\n5. LINEAR DISCRIMINANT ANALYSIS (LDA)")
    print("-" * 40)
    try:
        lda_analysis(dados, ixHealthy, ixCancer)
    except Exception as e:
        print(f"LDA analysis failed: {e}")
    
    print("\n" + "=" * 60)
    print("COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main()
