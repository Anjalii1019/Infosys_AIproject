import pandas as pd
import numpy as np

# Load user-item matrix
user_item_matrix = pd.read_csv(
    "data/user_item_matrix.csv",
    index_col=0
)

# Train-test split (80% train, 20% test)
def train_test_split(matrix, test_size=0.2):
    train = matrix.copy()
    test = pd.DataFrame(0, index=matrix.index, columns=matrix.columns)

    for user in matrix.index:
        non_zero_items = matrix.loc[user][matrix.loc[user] > 0].index.tolist()
        test_items = np.random.choice(
            non_zero_items,
            size=max(1, int(len(non_zero_items) * test_size)),
            replace=False
        )
        train.loc[user, test_items] = 0
        test.loc[user, test_items] = 1

    return train, test

train_matrix, test_matrix = train_test_split(user_item_matrix)

from sklearn.metrics.pairwise import cosine_similarity

# Compute similarity
user_similarity = cosine_similarity(train_matrix)
user_similarity_df = pd.DataFrame(
    user_similarity,
    index=train_matrix.index,
    columns=train_matrix.index
)

def recommend_top_k(user_id, k=5):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:]
    scores = train_matrix.loc[similar_users.index].T.dot(similar_users)
    scores = scores[train_matrix.loc[user_id] == 0]
    return scores.sort_values(ascending=False).head(k).index.tolist()

def evaluate_model(k=5):
    precision_list = []
    recall_list = []

    for user in test_matrix.index:
        actual_items = test_matrix.loc[user][test_matrix.loc[user] == 1].index.tolist()
        if not actual_items:
            continue

        recommended_items = recommend_top_k(user, k)

        true_positives = len(set(actual_items) & set(recommended_items))
        precision = true_positives / k
        recall = true_positives / len(actual_items)

        precision_list.append(precision)
        recall_list.append(recall)

    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    f1_score = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)

    return avg_precision, avg_recall, f1_score


precision, recall, f1 = evaluate_model()

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

#REFINEMENT

for k in [3, 5, 10]:
    p, r, f = evaluate_model(k)
    print(f"K={k} → Precision={p:.3f}, Recall={r:.3f}, F1={f:.3f}")
