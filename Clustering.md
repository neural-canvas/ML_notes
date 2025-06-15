similar records cluster -> groups - common properties/interest/values
![[Pasted image 20250506144650.png]]
`Birds of same feather flock together`

eg: Market Segmentation, Clustering in Pure Sciences

## **Similarity Algorithm**
Correlation Method
Distance Method

## **Types of Clustering Methods**

### Hierarchical Clustering

distance b/w two clusters
single linkage -shortest, complete linkage - longest, average linkage -avrerage
centroid distance - 2 cluster centroid distance

$$\frac{\sum{x_{i}}}{n}$$
Ward - ANOVA sum of squares b/w 2 clusters


| Method name                                                                                          | Parameters                 | Scalability                                                                                                                                  | Usecase                                                                             | Geometry (metric used)                       |
| ---------------------------------------------------------------------------------------------------- | -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | -------------------------------------------- |
| [K-Means](https://scikit-learn.org/stable/modules/clustering.html#k-means)                           | number of clusters         | Very large `n_samples`, medium `n_clusters` with [MiniBatch code](https://scikit-learn.org/stable/modules/clustering.html#mini-batch-kmeans) | General-purpose, even cluster size, flat geometry, not too many clusters, inductive | Distances between points                     |
| [Affinity propagation](https://scikit-learn.org/stable/modules/clustering.html#affinity-propagation) | damping, sample preference | Not scalable with n_samples                                                                                                                  | Many clusters, uneven cluster size, non-flat geometry, inductive                    | Graph distance (e.g. nearest-neighbor graph) |
| [Mean-shift](https://scikit-learn.org/stable/modules/clustering.html#mean-shift)                     | bandwidth                  | Not scalable with `n_samples`                                                                                                                | Many clusters, uneven cluster size, non-flat geometry, inductive                    | Distances between points                     |
| [Spectral clustering](https://scikit-learn.org/stable/modules/clustering.html#spectral-clustering)   | number of clusters         | Medium `n_samples`, small `n_clusters`                                                                                                       | Few clusters, even cluster size, non-flat geometry, transductive                    | Graph distance (e.g. nearest-neighbor graph) |


|                                                                                                                 |                                                                  |                                             |                                                                                                                |                                   |
| --------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| [Ward hierarchical clustering](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering) | number of clusters or distance threshold                         | Large `n_samples` and `n_clusters`          | Many clusters, possibly connectivity constraints, transductive                                                 | Distances between points          |
| [Agglomerative clustering](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering)     | number of clusters or distance threshold, linkage type, distance | Large `n_samples` and `n_clusters`          | Many clusters, possibly connectivity constraints, non Euclidean distances, transductive                        | Any pairwise distance             |
| [DBSCAN](https://scikit-learn.org/stable/modules/clustering.html#dbscan)                                        | neighborhood size                                                | Very large `n_samples`, medium `n_clusters` | Non-flat geometry, uneven cluster sizes, outlier removal, transductive                                         | Distances between nearest points  |
| [HDBSCAN](https://scikit-learn.org/stable/modules/clustering.html#hdbscan)                                      | minimum cluster membership, minimum point neighbors              | large `n_samples`, medium `n_clusters`      | Non-flat geometry, uneven cluster sizes, outlier removal, transductive, hierarchical, variable cluster density | Distances between nearest points  |
| [OPTICS](https://scikit-learn.org/stable/modules/clustering.html#optics)                                        | minimum cluster membership                                       | Very large `n_samples`, large `n_clusters`  | Non-flat geometry, uneven cluster sizes, variable cluster density, outlier removal, transductive               | Distances between points          |
| [Gaussian mixtures](https://scikit-learn.org/stable/modules/mixture.html#mixture)                               | many                                                             | Not scalable                                | Flat geometry, good for density estimation, inductive                                                          | Mahalanobis distances to centers  |
| [BIRCH](https://scikit-learn.org/stable/modules/clustering.html#birch)                                          | branching factor, threshold, optional global clusterer.          | Large `n_clusters` and `n_samples`          | Large dataset, outlier removal, data reduction, inductive                                                      | Euclidean distance between points |

|   |   |   |   |   |
|---|---|---|---|---|
|[Bisecting K-Means](https://scikit-learn.org/stable/modules/clustering.html#bisect-k-means)|number of clusters|Very large `n_samples`, medium `n_clusters`|General-purpose, even cluster size, flat geometry, no empty clusters, inductive, hierarchical|Distances between points|
## Agglomerative Hierarchical Method
sequencial - 

Single linkage
![[Pasted image 20250506151903.png]]

**threshold** - no of clusters are defined by threshold taken

![[Pasted image 20250506152546.png]]

**Silhouette Score** - best threshold value

**Dendogram plotting**
https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html#sphx-glr-auto-examples-cluster-plot-agglomerative-dendrogram-py

**Sklearn**
clust.labels_ gives all rows labels as encoded into 0, 1, 2

**Silhouette Score** - best threshold value
a,b,c,d
mean intra cluster distance (mean of ab, bc, bd) - A
mean nearest cluster distance - B

![[Pasted image 20250506161144.png]]

$$Sil(B) = \frac{B - A}{ max(A,B)}$$

Calculate sil coeff for all pts
Mean of all sil coeff

get threshold for which silhoette score is best, bigger the better

2 -- n labels -- n-1 records

gridsearchcv doesnot work

## Limitations of Hierarchical Clustering



![[Pasted image 20250506172026.png]]

## k-Means
https://www.naftaliharris.com/blog/visualizing-k-means-clustering/

**Algorhithm**
- Chose k random centroid
- reassign points
- Update centroid

rfm recency, frequency, monetary - data
https://www.techtarget.com/searchdatamanagement/definition/RFM-analysis

mini-batch kmeans

remove most recent days

![[Pasted image 20250506200302.png]]
### **Limitations of K-Means**
- Dependency on Initial Guess
- Sensitivity to Outliers
- Assumption of Round Clusters
- Need to Know the Number of Clusters
- Handling Large Datasets

## DBSCAN
**Density-Based Spatial Clustering of Applications with Noise**
k-Means - losely clustering - observations 
**Epsilon radius, minimum points**

https://ml-explained.com/blog/dbscan-explained

![[dbscan.gif]]

Reachability

## HDBSCAN


![[Pasted image 20250506202511.png]]

**Elbow Method** read it
inertia, sum of squares
WSS, 

RMSLE: 0.0601**388351584522**
RMSLE: 0.0601**1005603707664**
























