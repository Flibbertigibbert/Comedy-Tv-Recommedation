# Comedy-Tv-Recommedation
This is a practice Machine Learning Project by Zindi

## Introduction
A new Abuja-based online Comedy TV is seeking to optimise its content using customer ratings. Its contents are bought from popular comedy shows that are held in different parts of Nigeria. Therefore, it is important for the company to understand which content makes business sense based on customer ratings. In addition, it desires to offer personalised experience to each online viewer by delivering best comedy content that are most relevant to each viewer in its growing community of online and offline followers.

## Problem Statement 
To predict the ratings for some comedy events per individual based on the ratings by the same users for another set of comedy. This will inform its recommendation system.

## Approach Used

An hybrid model that combines three different prediction methods: SVD (Singular Value Decomposition), KNN (K-Nearest Neighbors), and LightGBM. Let's break it down:

1. SVD (Collaborative Filtering):
   - SVD is a matrix factorization technique used in collaborative filtering.
   - It works by decomposing the user-item interaction matrix into lower-dimensional user and item matrices.
   - SVD captures latent features that explain the observed ratings.

2. KNN (Memory-Based Collaborative Filtering):
   - KNN finds similar items (in our case, jokes) based on user rating patterns.
   - It predicts a user's rating for an item based on the ratings of similar items.

3. LightGBM (Machine Learning Model):
   - LightGBM is a gradient boosting framework that uses tree-based learning algorithms.
   - It takes features like average event rating and average user rating to make predictions.

Now, let's go through the process with an example:

Assume we have the following data:

User ID: U1001
Joke ID: J2005
Overall average rating for Joke J2005: 3.8
Overall average rating given by User U1001: 4.2

Step 1: SVD Prediction
```python
svd_pred = svd.predict('U1001', 'J2005').est
# Let's say this returns 3.9
```
SVD might predict 3.9 based on the latent features it has learned for this user and joke.

Step 2: KNN Prediction
```python
knn_pred = knn.predict('U1001', 'J2005').est
# Let's say this returns 4.1
```
KNN might predict 4.1 based on how similar users rated this joke or how this user rated similar jokes.

Step 3: LightGBM Prediction
```python
event_avg = 3.8  # Average rating for Joke J2005
user_avg = 4.2   # Average rating given by User U1001
lgb_pred = lgb_model.predict([[event_avg, user_avg]])[0]
# Let's say this returns 4.0
```
LightGBM predicts 4.0 based on the average ratings of the joke and the user.

Step 4: Combine Predictions
```python
hybrid_pred = (0.4 * svd_pred) + (0.3 * knn_pred) + (0.3 * lgb_pred)
hybrid_pred = (0.4 * 3.9) + (0.3 * 4.1) + (0.3 * 4.0)
hybrid_pred = 1.56 + 1.23 + 1.2 = 3.99
```

The final prediction is a weighted average of the three individual predictions. The weights (0.4, 0.3, 0.3) can be adjusted based on which model performs best.

Explanation of the Hybrid Approach:

1. SVD captures global patterns in the data, understanding overall user preferences and joke characteristics.
2. KNN captures local patterns, considering similar users or jokes.
3. LightGBM captures non-linear relationships between the average ratings and the actual ratings.

By combining these methods, we aim to leverage the strengths of each:
- SVD is good at handling sparsity and capturing latent features.
- KNN can capture very specific local preferences.
- LightGBM can capture complex relationships and is good at handling non-linear patterns.

The hybrid model can potentially outperform any single model because:
- If one model makes a poor prediction, the others can compensate.
- Each model captures different aspects of the data, providing a more comprehensive prediction.

In our example:
- SVD predicted slightly lower than the joke's average, possibly because the user's latent features suggest they might not enjoy this type of joke as much.
- KNN predicted higher, possibly because similar users enjoyed this joke.
- LightGBM predicted in between, balancing the user's tendency to rate high with the joke's slightly lower average rating.

The final prediction of 3.99 takes all these factors into account, providing a balanced estimate that's slightly higher than the joke's average (reflecting the user's tendency to rate high) but not as high as the user's average rating (reflecting that this joke isn't rated as highly as the user's typical ratings).

This approach allows for nuanced predictions that consider multiple factors and modeling techniques, potentially leading to more accurate and robust recommendations.
