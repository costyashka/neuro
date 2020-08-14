from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt 
from mglearn import discrete_scatter

X,y = make_blobs(centers=8,cluster_std=2000.0,n_samples=350)
model = KMeans(n_clusters=8)
model.fit(X,y)

discrete_scatter(X[:,0],X[:,1],model.labels_,markers='o')
discrete_scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1],[0,1,2,3,4,5,6,7],markers='^',
                markeredgewidth=2)
plt.show()
