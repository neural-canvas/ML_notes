
## Non Personalised Recommendations

## Personalised Recommendations
- Content Based Filtering
- User Based Collaborative Filtering UBCF

### User Based Collaborative Filtering UBCF
https://www.analyticsvidhya.com/blog/2022/02/introduction-to-collaborative-filtering/
cosine similarity

$$cosine \ similarity(A,B) = \frac{A.B}{|A||B|} = \frac{\sum^{n}_{i=1}A_{i}B_{i}}{\sqrt{ \sum^{n}_{i=1}A_{i}^{2} }\sqrt{ \sum^{n}_{i=1}B_{i}^{2} }}$$

**Average expected rating**
$$Exp \ Rating = 
\frac{\sum r_{i}s_{i}}{\sum s_{i}}$$
Need for Normalisation
Collaborative filtering with binary data
Jacqard Index - Intersection over union (IoU)

![[Pasted image 20250509195213.png]]
https://www.encora.com/insights/recommender-system-series-part-2-neighborhood-based-collaborative-filtering

**Behroj Biryani**

## Surprise library
pip install scikit-surprise

need **Sparse Matrix Format**
long 

Dataset for Recommendation System
https://grouplens.org/
https://grouplens.org/datasets/movielens/
