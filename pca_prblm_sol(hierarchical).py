import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
data = pd.read_csv(r'E:\ExcelR ass\pca\wine.csv')

# avoid pre_clustered column as per problem statement :
df = data.iloc[:,1:]
# print(df.head())
# print(df.shape)        # (178, 13)

# apply normalisation :
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)
df_norm = norm_func(df)    # (178, 13)

#apply PCA :
pcs = PCA(n_components=13)
pcs_values = pcs.fit_transform(df_norm)
pcs_df = pd.DataFrame(pcs_values)
final_df = pcs_df.iloc[:,0:3]
# print(final_df.head())

# plot Dendrogram to get optimum number of clusters :
# dendogram = sch.dendrogram(sch.linkage(final_df, method="complete"))
# plt.show()
# from above dendogram , we have to take 4 clusters which is not same as origianl data :

hc = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='complete')
y_hc = hc.fit_predict(df_norm)
# print(y_hc)

# plot hierarchical clusters :
final_df["h_clustered"] = pd.Series(y_hc)
# plt.scatter(final_df[0].loc[final_df.h_clustered==0], final_df[1].loc[final_df.h_clustered==0])
# plt.scatter(final_df[0].loc[final_df.h_clustered==1], final_df[1].loc[final_df.h_clustered==1])
# plt.scatter(final_df[0].loc[final_df.h_clustered==2], final_df[1].loc[final_df.h_clustered==2])
# plt.scatter(final_df[0].loc[final_df.h_clustered==3], final_df[1].loc[final_df.h_clustered==3])
# plt.show()

# there for, we can conclude that numbers of hierarchial clusters obtained after applying PCA is not same as given in raw data
