**Types**
1. point
2. collective
3. contextual

Outliers Detection

- Box Whisker method, DBSCAN
- Isolation Forest

## Isolation Forest

isolating them within a forest of random decision trees
keys: Random partitioning, tree structure, anomaly detection, ensemble trees
working: forest - random data subset, random splitting - random feature subset, path length - equivalent to isolation of pts, **anomaly score**- Lower scores indicate a higher probability of being an anomaly. 


https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
https://www.geeksforgeeks.org/what-is-isolation-forest/


contamination = 0.05 ( <= 0.05)

![[Pasted image 20250507142348.png]]

**Q. Outlier removal and applying random forest**

