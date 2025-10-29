import scipy.stats as stats

#---------Kruskal-Wallis H-test---------
def kruskal_wallis_test(dados, ixHealthy, ixCancer, fnames):
    H_rank=[]

    for feature in fnames:
        H,p= stats.kruskal(dados[feature].iloc[ixHealthy],
                        dados[feature].iloc[ixCancer])
        H_rank.append((feature,H))

    print("Kruskal-Wallis H-test results (feature, H-statistic):")

    Hs=sorted(H_rank, key=lambda x: x[1], reverse=True)

    for f in Hs:
        print(f"Feature: {f[0]}, H-statistic: {f[1]:.4f}")

    
    return Hs
