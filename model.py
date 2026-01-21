import pandas as pd

user_item_matrix = pd.read_csv(
    "data/user_item_matrix.csv",
    index_col=0
)

print(user_item_matrix.shape)

from sklearn.metrics.pairwise import cosine_similarity

user_similarity = cosine_similarity(user_item_matrix)

user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_item_matrix.index,
    columns=user_item_matrix.index
)

print(user_similarity_df.head())

def recommend_products(user_id, num_recommendations=5):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:]
    
    weighted_scores = user_item_matrix.loc[similar_users.index].T.dot(similar_users)
    
    already_rated = user_item_matrix.loc[user_id]
    weighted_scores = weighted_scores[already_rated == 0]
    
    return weighted_scores.sort_values(ascending=False).head(num_recommendations)

sample_user = user_item_matrix.index[0]
recommendations = recommend_products(sample_user)

print("Recommended products:")
print(recommendations)

coverage = (user_item_matrix > 0).sum().sum() / user_item_matrix.size
print("Coverage:", coverage)

sparsity = 1 - coverage
print("Sparsity:", sparsity)

user_similarity_df.to_csv("data/user_similarity_matrix.csv")