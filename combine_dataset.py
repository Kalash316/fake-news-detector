import pandas as pd

# Step 1: Load both datasets
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

# Step 2: Add label columns
fake_df["label"] = "FAKE"
true_df["label"] = "REAL"

# Step 3: Combine both into one dataframe
df = pd.concat([fake_df, true_df])

# Step 4: Shuffle (optional but recommended)
df = df.sample(frac=1).reset_index(drop=True)

# Step 5: Save to news.csv
df.to_csv("news.csv", index=False)

print("✅ news.csv created successfully!")
