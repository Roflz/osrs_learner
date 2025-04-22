import pandas as pd

df = pd.read_csv("data/xp_events.csv")

# Extract numeric XP value
df["xp_value"] = df["xp_text"].str.extract(r"(\\d+)").astype(float)

# Encode categorical features
df["action_id"] = df["action_type"].astype("category").cat.codes
df["xp_type_id"] = df["xp_type"].astype("category").cat.codes

# Save cleaned training data
df.to_csv("data/clean_xp_data.csv", index=False)
print("Saved cleaned dataset to data/clean_xp_data.csv")
