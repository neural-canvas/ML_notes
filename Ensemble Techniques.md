**combine** the predictions from two or more model fits - similar or dissimilar types called base learners or weak learners.

system of predictive modelling more **robust** than the individual algorithms.

## Principle
Categorical Predictions - Majority Vote
Numerical Predictions - Averaging, Weighted Averaging

## Types
- Voting
- Bagging
- Boosting
- Stacking

### Voting
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html#sklearn.ensemble.VotingClassifier
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html#sklearn.ensemble.VotingRegressor
#### Classification

```python
eclf1 = VotingClassifier(estimators=[
        ('svm-rbf', clf1), ('svm-linear', clf2), ('k-NN', clf3)], voting='hard')
```

**voting='hard'/'soft'**
Hard voting relies on the **majority vote** of the predicted class labels, while 
Soft voting considers the **predicted probabilities** and uses a **weighted average** to determine the final class
#### Regression
mean of prediction of estimators
```python
er = VotingRegressor([('lr', r1), ('rf', r2), ('r3', r3)])
```
### Bootstrap Aggregating - Bagging
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html#sklearn.ensemble.BaggingRegressor

Multiple bootstrapped **samples** are drawn from the same data with **replacement** - fit the **same** model - majority vote or averaging to get the final **prediction**.
eg: Random Forest algorithm
```python
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=100, n_features=4,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)

clf = BaggingClassifier(estimator=SVC(),
                        n_estimators=10, random_state=0).fit(X, y)

clf.predict([[0, 0, 0, 0]])
```

usually better performance with decision tree 
### Random Forest
bagging - estimators decision trees - max features
it is a decision tree but with an extra feature - max features- this means max features to consider while splitting in tree

```python
```
### Boosting
**sequential** technique of fitting algorithms - on **residuals** - weak learners - reduces the bias - use parameter tuning..avoiding **over-fitting** 
eg: ADABOOST, GBM, XGBOOST,  LightGBM

Algos
1. Add weight to each training data -> train weak learners -> eg weight as 1
2. Rescale misclassification -> add more weights to incorrect pred from 1st learner -> eg weight as 2.33 -> update evaluation metrics
3. Repeat
4. Voting -> Weighted Voting -> weights as ratio of success/fail

https://www.youtube.com/watch?v=AtYN8QP-U6w

http://rob.schapire.net/papers/explaining-adaboost.pdf
several **different approaches** that have been proposed for understanding AdaBoost

Xtrain - upweighting error sampled datapoints in training data

### **Types**
Adaptive Boosting
eg: ADABOOST

```python
from sklearn.ensemble import AdaBoostClassifier
clf1 = AdaBoostClassifier(n_estimators=100)
```

#### Params
**estimator**, default=None
If None, then the base estimator is DecisionTreeClassifier, initialized with max_depth=1

### Gradient Boosting
GBM, XGBOOST,  LightGBM
generalization of Ada Boost - gradient descent + boosting

https://www.youtube.com/watch?v=jxuNLH5dXCs 

**sklearn implementation**
GradientBoostingClassifier
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html

Params

| params        | desc                             | range    |
| ------------- | -------------------------------- | -------- |
| n_estimators  | number of boosting stages        | 1, inf   |
| max_depth     | individual regression estimators | 1, inf   |
| learning_rate |                                  | 0.0, inf |

**xgboost implementation** - Tianqi Chen
https://xgboost.readthedocs.io/en/release_3.0.0/

**lightgbm implementation** - Microsoft
https://lightgbm.readthedocs.io/en/stable/
pip install lightgbm

**catboost implementation** - Yandex
https://catboost.ai/docs/en/concepts/python-installation
https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier

### Stacking
Stack of estimators with a final classifier.
similar to the voting ensembles but also **assigns the weights** to the machine learning algorithms - ground models and meta models - best
meta models - assign different **weights** to the ground models

**outputs** of the ground models on the test data as the **training data** for the meta-model

```python
clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=11
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)
```

## Miscellaneous
stump: max depth tree

## Further Reading
https://www.youtube.com/@statquest
https://www.deeplearning.ai/short-courses/attention-in-transformers-concepts-and-code-in-pytorch/

Bias
Variance
Bias Variance Trade-Off

![[Pasted image 20250505152453.png]]


![[Pasted image 20250505152619.png]]

Overfitting and Underfitting



