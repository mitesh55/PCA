from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
data = pd.read_csv(r'E:\ExcelR ass\pca\wine.csv')
# print(data.head())
# print(data.shape)      # (178, 14)

# as per statement we are having 3 cluster but for pca calculation we're gonna ignore type column:
df = data.iloc[:,1:]
# print(df.head())
# print(df.shape)        # (178, 13)

# apply PCA on normalized data :
pcs = PCA(n_components=13)
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)

df_norm = norm_func(df)
# print(df_norm.head())
pcs_values = pcs.fit_transform(df_norm)
pcs_df = pd.DataFrame(pcs_values)
# print(pcs_df.head())
var = (pcs.explained_variance_ratio_)*100
cumvar = np.cumsum(np.round(var, decimals=4))
# print(cumvar)


# now we're retriving first 3 pcs as per problem statement :  (68.2816 %)
final_pcs_df = pcs_df.iloc[:,0:3]
# print(final_pcs_df.head())
kmeans = KMeans(n_clusters=3)
kmeans.fit(final_pcs_df)
kmeans_labels = kmeans.labels_
cluster_center = kmeans.cluster_centers_
# print(cluster_center)
final_pcs_df["k_clustered"] = pd.Series(kmeans_labels)
# print(final_pcs_df.head())
# print(kmeans_labels)

# Elbow graph :
k = list(range(2,7))
twss = []

for i in k:
    loop_kmeans = KMeans(n_clusters=i)
    loop_kmeans.fit(final_pcs_df)
    wss = []
    for j in range(i):
        wss.append(sum(cdist(final_pcs_df.iloc[loop_kmeans.labels_==j], loop_kmeans.cluster_centers_[j].reshape(1, final_pcs_df.shape[1]), 'euclidean')))
    twss.append(sum(wss))
# plt.plot(k, twss, 'ro-')
# plt.show()
# from above Elbow graph it is clear that optimum k value is 3.


# graph for original data (Without applying PCA):
new_df = pcs_df.iloc[:,0:2]
new_df["pre_clustered"] = data['Type']
# print(new_df.head())
# plt.scatter(new_df[0].loc[new_df.pre_clustered==1], new_df[1].loc[new_df.pre_clustered==1])
# plt.scatter(new_df[0].loc[new_df.pre_clustered==2], new_df[1].loc[new_df.pre_clustered==2])
# plt.scatter(new_df[0].loc[new_df.pre_clustered==3], new_df[1].loc[new_df.pre_clustered==3])
# plt.title("Raw Clustered Data")
# plt.legend()
# plt.show()



# Graph of data after applying PCA :
# plt.scatter(final_pcs_df[0].loc[kmeans_labels==0], final_pcs_df[1].loc[kmeans_labels==0], c="blue")
# plt.scatter(final_pcs_df[0].loc[kmeans_labels==1], final_pcs_df[1].loc[kmeans_labels==1], c="green")
# plt.scatter(final_pcs_df[0].loc[kmeans_labels==2], final_pcs_df[1].loc[kmeans_labels==2], c="orange")

# for Cluster center point :
# plt.scatter(cluster_center[0][0], cluster_center[0][1], marker='*', s=50, c="black")
# plt.scatter(cluster_center[1][0], cluster_center[1][1], marker='*', s=50, c="black")
# plt.scatter(cluster_center[2][0], cluster_center[2][1], marker='*', s=50, c="black")
# plt.legend()
# plt.title("CLUSTER USING PCA")
# plt.show()


# there for we can say that, number of cluster from original data is same as derived clusters(KMeans) using principal component analysis.