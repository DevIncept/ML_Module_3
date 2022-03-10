# CLUSTERING
Its Unsupervised Learning, providing insights on categories and groups that the dataset can fit into... 
![](https://i.pinimg.com/originals/ce/f2/18/cef218767880fc469a41ca739b0a2539.jpg)

## INTRODUCTION:
it's essentially a sort of unsupervised learning methodology . associate unsupervised learning method may be a method within which we have a tendency to draw references from datasets consisting of input file while not tagged responses. Generally, it is used as a method to seek out meaningful structure, informative underlying processes, 
generative features, and clusterings inherent in an exceedingly set of examples.
Bunch is that the task of dividing the population or data points into variety of teams specified data points within the same groups are a lot of similar to different data points in the same group and dissimilar to the information points in different groups. It's essentially a set of objects on the idea of similarity and difference between them.

## Type of Algorithms
1. K -mean
2. Affinity propagation
3. Mean Shift
4. Spectral Clustering
5. Hierarchical clustering
6. DBSCAN
7. OPTICS
8. Guassian Matrix
9. BIRCH

### K-means
K-means is often called as Lloyd’s algorithm. It consist of few steps:

- chooses the initial centroids [Choosing K from X]
And then...
- The first assigns each sample to its nearest centroid.
- The second step creates new centroids by taking the mean value of the existing centroids.
- And the rest is looping between the above steps

This step continues untill the new centroids are nearly the same as the old ones.
Other types of K-means:
 1. Low-level parallelism
 2. Mini K-means

### Affinity Propagation
In Affinity Propagation there is no need to specify the number of clusters. Its more suitable where we don't know thw optimal number of clusters. The interesting thing about this machine learning techniques is that you don’t have to configure the number of clusters in advance. 
The algorithm has a high time complexity, which is a demerit of using it.
- It creates clusters by sending messages between pairs of samples until convergence.


```
from sklearn.datasets import make_blobs

X, y_true = make_blobs(n_samples=300, centers=4,  n_features = 2, cluster_std=0.60)

plt.scatter(X[:, 0], X[:, 1], s=50);
```

![](https://i.pinimg.com/originals/34/ef/3b/34ef3b985eef47885ad7e832075ef9d1.jpg)

```
af = AffinityPropagation(preference=-40)
clustering = af.fit(X)
plt.scatter(X[:,0], X[:,1], c=clustering.labels_, cmap='rainbow', alpha=0.7, edgecolors='r')
```
![](https://i.pinimg.com/originals/4e/1e/83/4e1e834616277078ddc305a23bfec29d.jpg)


### Mean shift
The clustering aims on finding blobs in a smooth density of samples present. 
- Centroids with mean of the given region
- They are revised to remove the duplicates, thus increasing the efficiency of cluster formation.

The algorithm automatically sets the number of clusters, instead of relying on a parameter bandwidth, which dictates the size of the region to search through.
The algorithm is not highly scalable, as it requires multiple nearest neighbor searches during the execution of the algorithm.
Again the algorithm stops iteration if the change is no more significant.

```
ms = MeanShift()
ms.fit(X)
cluster_centers = ms.cluster_centers_
plt.scatter(X[:,0], X[:,1], c=clustering.labels_, cmap='rainbow', alpha=0.7, edgecolors='r')
```
The results will be similar to affinity propagations.

### Spectral Clustering
- low-dimension embedding of the affinity matrix between samples, followed by clustering
The present version requires the number of clusters to be specified in advance. It works well for a small number of clusters, but is not advised for many clusters.

Spectral clustering is a technique with roots in graph theory, where the approach is used to identify communities of nodes in a graph based on the edges connecting them.

```
spectral_model_rbf = SpectralClustering(affinity ='rbf')
  
# Training the model and Storing the predicted cluster labels
labels_rbf = spectral_model_rbf.fit_predict(X)
```

### Heirarchical Clustering
Building nested clusters by merging or splitting them successively.Formation of hierarchy of clusters. 
At first all the points are considerded as diffirent clusters. then they are driven to combine and form hierarchy of clusters.

1. Agglomerative Clustering
Scale to large number of samples when it is used jointly with a connectivity matrix, but is computationally expensive when no connectivity constraints are added between samples: it considers at each step all the possible merges.

2. Divisive Clustering
Is the exact opposite of Agglomerative.

### DBSCAN
Density-based spatial clustering of applications with noise (DBSCAN).
Clusters as areas of high density separated by areas of low density.
Clusters found by DBSCAN can be any shape, as opposed to k-means which assumes that clusters are convex shaped.
- DBSCAN groups together points that are close to each other based on a distance measurement (usually Euclidean distance)

### OPTICS
Ordering points to identify the clustering structure(OPTICS).
The most common density based approach, DBSCAN, requires only two parameters.
The OPTICS algorithm shares many similarities with the DBSCAN algorithm. But its different in some parameters. It relaxes some requirements. 

### BIRCH
Balanced Iterative Reducing and Clustering using Hierarchies(BIRCH)
The Birch builds a tree called the Clustering Feature Tree (CFT) . The CF Nodes have a number of subclusters called Clustering Feature subclusters (CF Subclusters) and these CF Subclusters located in the non-terminal CF Nodes can have CF Nodes as children.

The BIRCH algorithm has two parameters, the threshold and the branching factor. The branching factor limits the number of subclusters in a node and the threshold limits the distance between the entering sample and the existing subclusters.
```
brc = Birch(branching_factor=50, n_clusters=None, threshold=1.5)
brc.fit(X)
```

### Guassian Mixture
Gaussian Mixture Models assume that there are a certain number of Gaussian distributions, and each of these distributions represent a cluster.


## APPLICATIONS OF CLUSTERING ALGORITHMS

![](https://i.pinimg.com/originals/3f/fb/47/3ffb4706ce0c6be3cf6ca51c8759f89f.jpg)

1. Clustering algorithm can be used for identifying fake news.

It works by collecting content of fake news article, or you can say corpus,then make cluster of words. These clusters will help the algorithm to predict which news are fake and which news are not fake. 

2. Clustering Algorithms  can be used in Search Engines:

Search engines uses clustering algorithm to group similar objects together that is objects related to data you search on search engine and by grouping dissimilar objects together . So, when you search, you also get some other objects related to your search. 

3. K-means algorithm can be  used for  image compression, market segmentation, document clustering and image segmentation. 

4. The clustering algorithm can be used for the identification of cancer cells. It works by making clusters  of cancerous data and non-cancerous data. 

5.  The clustering algorithm can be used in Biology. It works by using  image recognition and make clusters of different species of plants and animals. 

6. K-means clustering techniques can be used to identify spam emails.
7.Clustering algorithm helps us by predicting earthquake - affected areas and by grouping some dangerous zones and less dangerous zones.

### Misc...
Examplar model notebook link
https://www.kaggle.com/prasanshasatpathy/clustering-unsupervised













