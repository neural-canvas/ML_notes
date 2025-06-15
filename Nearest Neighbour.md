Date- 29/4/25
Day-Tuesday
k-nearest neighbors vote

The principle behind nearest neighbor methods is to find a predefined number of training samples closest in distance to the new point, and predict the label from these

fast indexing structure such as a 
1. brute-force
2. Ball Tree
3. KD Tree

computational inefficiencies - tree-based data structures

params
1. n_neighbors
2. metric
3. 


## Distance
**Minkowski distance** is a generalized measure of distance, encompassing both Euclidean and Manhattan distances as special cases

**Euclidean distance** has bias towards larger values
using scaling may help

## Scaling
https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#plot-all-scaling-standard-scaler-section

### Standard Scaling
$$Z = \frac{x- \mu}{\alpha}$$


### Min Max Scaling
$$Z = \frac{X - min(X)}{max(X)-min(X)}$$
range - (0, 1)

### Others
- MaxAbsScaler,
- **MinMaxScaler**,
- Normalizer,
- PowerTransformer,
- QuantileTransformer,
- RobustScaler,
- **StandardScaler**,
- minmax_scale


![[Pasted image 20250429222754.png]]
# Explanation of the Bernoulli Naive Bayes Algorithm

This is an implementation of the Bernoulli Naive Bayes (NB) classifier, which is particularly suited for binary document classification tasks (like spam detection) where the presence or absence of words matters more than their frequency.

## Training Phase (TRAINBERNOULLINB)

1. **Vocabulary Extraction**: Line 1 extracts all unique terms (words) from the training documents to create vocabulary V.

2. **Document Counting**: Line 2 counts the total number of training documents (N).

3. **Class Processing**: For each class c (lines 3-8):
   - Count how many documents belong to this class (N_c)
   - Calculate the prior probability for the class (prior[c]) as the proportion of documents in this class
   - For each term in the vocabulary:
     - Count how many documents in this class contain the term (N_ct)
     - Calculate the conditional probability P(t|c) with add-one smoothing: (N_ct + 1)/(N_c + 2)

The smoothing prevents zero probabilities for terms not seen in a class during training.

## Classification Phase (APPLYBERNOULLINB)

1. **Term Extraction**: Line 1 extracts terms from the test document that are in the vocabulary.

2. **Scoring**: For each class (lines 2-7):
   - Start with the log of the class prior probability
   - For each vocabulary term:
     - If the term appears in the document: add log(P(t|c))
     - If the term doesn't appear: add log(1 - P(t|c))
   
3. **Decision**: Return the class with the highest score (line 8).

## Key Characteristics

- **Bernoulli Model**: Treats each document as a binary vector over the vocabulary (1 if term present, 0 if absent)
- **Smoothing**: Uses Laplace/add-one smoothing to handle unseen terms
- **Log Probabilities**: Uses logarithms to avoid underflow with many small probabilities multiplied together

This algorithm is efficient and works well for many text classification problems, especially when word presence/absence is more important than frequency.

**Assumptions**
normal - features
independent - features

CategoricalNB
https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.CategoricalNB.html#sklearn.naive_bayes.CategoricalNB

GaussianNB
https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB

https://scikit-learn.org/stable/modules/naive_bayes.html