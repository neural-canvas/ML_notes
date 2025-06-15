Date - 1/5/25
Day - Thursday

https://www.kaggle.com/code/prashant111/decision-tree-classifier-tutorial
https://www.analyticsvidhya.com/blog/2021/08/decision-tree-algorithm/
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
https://scikit-learn.org/stable/modules/tree.html
https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py

Decision Trees (DTs) are a **non-parametric** **supervised learning method** used for classification and regression. The goal is to create a model that predicts the value of a target variable by **learning simple decision rules** inferred from the data features. A tree can be seen as a **piecewise constant approximation**.

how good each column is
1. Gini
2. Entropy and Information Gain

## Decision Tree Assumptions 
tree’s construction and impact its performance
Binary Splits,
Recursive Partitioning,
Feature Independence,
Homogeneity - achieving clear decision boundaries,
Top-Down Greedy Approach - split is chosen to maximize information gain or minimize impurity at the current node

## Advantages of Decision Trees
- Simple to understand and to interpret. Trees can be visualized.
- Requires little data preparation. Other techniques often require data normalization, dummy variables need to be created and blank values to be removed. Some tree and algorithm combinations support [missing values](https://scikit-learn.org/stable/modules/tree.html#tree-missing-value-support).
- The cost of using the tree (i.e., predicting data) is logarithmic in the number of data points used to train the tree.
- Able to handle 
- both numerical and categorical data. However, the scikit-learn implementation does not support categorical variables for now. Other techniques are usually specialized in analyzing datasets that have only one type of variable. See [algorithms](https://scikit-learn.org/stable/modules/tree.html#tree-algorithms) for more information.
- Able to handle multi-output problems.
- Uses a white box model. If a given situation is observable in a model, the explanation for the condition is easily explained by boolean logic. By contrast, in a black box model (e.g., in an artificial neural network), results may be more difficult to interpret.
- Possible to validate a model using statistical tests. That makes it possible to account for the reliability of the model.
- Performs well even if its assumptions are somewhat violated by the true model from which the data were generated.

## Disadvantages of Decision Trees
- Decision-tree learners can create over-complex trees that do not generalize the data well. This is called overfitting. Mechanisms such as pruning, setting the minimum number of samples required at a leaf node or setting the maximum depth of the tree are necessary to avoid this problem.
- Decision trees can be unstable because small variations in the data might result in a completely different tree being generated. This problem is mitigated by using decision trees within an ensemble.
- Predictions of decision trees are neither smooth nor continuous, but piecewise constant approximations as seen in the above figure. Therefore, they are not good at extrapolation.
- The problem of learning an optimal decision tree is known to be NP-complete under several aspects of optimality and even for simple concepts. Consequently, practical decision-tree learning algorithms are based on heuristic algorithms such as the greedy algorithm where locally optimal decisions are made at each node. Such algorithms cannot guarantee to return the globally optimal decision tree. This can be mitigated by training multiple trees in an ensemble learner, where the features and samples are randomly sampled with replacement.
- There are concepts that are hard to learn because decision trees do not express them easily, such as XOR, parity or multiplexer problems.
- Decision tree learners create biased trees if some classes dominate. It is therefore recommended to balance the dataset prior to fitting with the decision tree.

## Types of Decision Tree
ID3, CART- Classification And Regression Tree
https://www.geeksforgeeks.org/cart-classification-and-regression-tree-in-machine-learning/
### CART
keys: Tree structure, Splitting criteria, Pruning
- supervised learning algorithm
- This algorithm uses a different measure called Gini impurity to decide how to split the data
- It can be used for both classification (sorting data into categories) and regression (predicting continuous values) tasks

classification tasks -> CART uses Gini impurity
regression tasks -> CART uses residual reduction as the splitting criterion

Cost Function (Minimises)
$$J(k, t_{k}) = \frac{m_{left}}{m}G_{left} + \frac{m_{right}}{m}G_{right}$$
where 
m: no of observation
G: gini index

implementation: all values, percentiles, deciles, quantiles values only

#### Gini Impurities

$$\sum_{i}f_{i}(1- f_{i})$$
$$ = f_{i}(1-f_{i}) + f_{2}(1-f_{2})$$

e:g
Value 20, 30 ; gini = 0.48

child nodes - gini index is lower than parent node, gini index = zero does not have child node

## The Idea of Entropy  and Information Gain

https://www.analyticsvidhya.com/blog/2023/01/step-by-step-working-of-decision-tree-algorithm/

## Code

```python
clf = DecisionTreeClassifier(random_state=25,
	max_depth=d,
	min_samples_split=s
	)
```

### params
- max_depth: maximum depth of the tree 
- min_samples_split: The minimum number of samples required to split an internal node - helps in pruning
- min_samples_leaf: minimum number of samples required to be at a leaf node - helps in pruning

## Tree
```python
plt.figure(figsize=(35, 15))
plot_tree(best_model, feature_names=list(X.columns),
          class_names=['Benign', 'Malignant'],
          filled=True,
          fontsize=14)
plt.show()
```

![[Pasted image 20250501182144.png]]



## Regression Tree

https://scikit-learn.org/stable/modules/tree.html#tree
https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html
https://scikit-learn.org/1.6/modules/generated/sklearn.tree.DecisionTreeRegressor.html

mse

$$sqerr = \frac{\sum (y_{i} - value)^{2}}{samples}$$















