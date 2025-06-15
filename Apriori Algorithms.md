https://rasbt.github.io/mlxtend/api_subpackages/mlxtend.frequent_patterns/

Association mining aims to learn patterns/substructure/knowledge from a dataset through association rules. 

For example, we can use transaction records of a supermarket to investigate which items are often **bought together**.

**Rules**
Data -> Apriori Algorithms -> Frequent item sets -> Rules

**Examples**
Sample Customers's Basket -> Generate Rules -> Recommendation

### Association Rules
the pattern/co-occurrence of two item sets by using an if-then condition. 

For example, a rule (Apple) → (Banana) means “IF Apple is in a transaction, THEN Banana is also in that transaction”.

### Strength of Association
- Support
- Confidence
- Lift Ratio

![[Pasted image 20250509141208.png]]
### Support
$$\text{support}(A\rightarrow C) = \text{support}(A \cup C), \;\;\; \text{range: } [0, 1]$$
frequency with which an item-set (a collection of items) appears in a dataset
as a **probability** (normalised value)

if-part (Antecedent) --> then-part (Consequent)

Examples:
count {{milk, sugar} -> {tea}} / N



### Confidence
$$\text{confidence}(A\rightarrow C) = \frac{\text{support}(A\rightarrow C)}{\text{support}(A)}, \;\;\; \text{range: } [0, 1]$$


trustworthiness - discovered rule


$$Confidence = \frac{Support(if - part \ and \ then - part)}{Support(if-part)} $$

confidence as probability
$$P(A \cap B ) = P(B|A) P(A)$$
$$P(B | A ) = \frac{P(A \cap B)}{P(A)}$$

**Independence**
if-part and then-part is independence
$$Confidence (P(then - part | if-part)) = Confidence(P(then - part))$$
**Benchmark Confidence**
No of transaction of then part / Total transaction
if-part and then-part is not independence
Conf > Benchmark

### Lift Ratio
$$\text{lift}(A\rightarrow C) = \frac{\text{confidence}(A\rightarrow C)}{\text{support}(C)}, \;\;\; \text{range: } [0, \infty]$$
measures the strength of the **relationship** between an antecedent and a consequent
**lift ratio > 1** -> association rule has some usefulness
$$lift \ ratio = \frac{Conf}{B. Conf} > 1$$
### Interpreting the Results
https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/#association_rules-association-rules-generation-from-frequent-itemsets
### Code Snippets
https://www.kaggle.com/code/rockystats/apriori-algorithm-or-market-basket-analysis/comments#3198107

```python
# Encode Data
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)

# Apriori
from mlxtend.frequent_patterns import apriori
apriori(df, min_support=0.6)
```
https://rpubs.com/XH8851/883280