import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load, scale, and cluster in 3 steps
x = StandardScaler().fit_transform(load_breast_cancer().data)
km = KMeans(2, random_state=42, n_init='auto').fit(x)

# Plotting
plt.scatter(x[:, 0], x[:, 1], c=km.labels_, cmap='viridis')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], c='red', marker='x', s=200,label='Centroids')
plt.title('K-Means Clustering'); 
plt.legend()
plt.show()
