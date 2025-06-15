## One hot encoding
Nominal Categorical data
```python
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
X = [['Male', 1], ['Female', 3], ['Female', 2]]
enc.fit(X)
enc.categories_
enc.transform([['Female', 1], ['Male', 4]]).toarray()
enc.inverse_transform([[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]])
enc.get_feature_names_out(['gender', 'group'])

drop_enc = OneHotEncoder(drop='first').fit(X)
drop_enc.categories_
drop_enc.transform([['Female', 1], ['Male', 2]]).toarray()
```


## Ordinal Encoder

```python
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
X = [['Male', 1], ['Female', 3], ['Female', 2]]
enc.fit(X)
enc.categories_
enc.transform([['Female', 3], ['Male', 1]])
```

Maps categorical array into integer array

## Label Encoder
for y -> multi class classification and y is categorical

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit([1, 2, 2, 6])
le.classes_
le.transform([1, 1, 2, 6])
le.inverse_transform([0, 0, 1, 2])

le = LabelEncoder()
le.fit(["paris", "paris", "tokyo", "amsterdam"])
list(le.classes_)
le.transform(["tokyo", "tokyo", "paris"])
list(le.inverse_transform([2, 2, 1]))
```

