import pandas as pd

# Load datasets
orders = pd.read_csv("data/olist_orders_dataset.csv")
order_items = pd.read_csv("data/olist_order_items_dataset.csv")
reviews = pd.read_csv("data/olist_order_reviews_dataset.csv")

# Merge datasets
df = reviews.merge(orders, on="order_id", how="inner")
df = df.merge(order_items, on="order_id", how="inner")

interactions = df[["customer_id", "product_id", "review_score"]]

# 🔥 LIMIT SIZE (VERY IMPORTANT)
interactions = interactions.sample(20000, random_state=42)

# Create User–Item Matrix
user_item_matrix = interactions.pivot_table(
    index="customer_id",
    columns="product_id",
    values="review_score",
    fill_value=0
)

user_item_matrix.to_csv("data/user_item_matrix.csv")

print("✅ User–Item Matrix created (sampled)")
print(user_item_matrix.shape)
print(user_item_matrix.head())
