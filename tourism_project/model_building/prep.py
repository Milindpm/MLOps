
"""
Data Preparation Script
Author: Milind Mundankar
"""

import os
import pandas as pd

from sklearn.model_selection import train_test_split


# ----------------------------------------------------------
# Paths
# ----------------------------------------------------------

DATA_PATH = "tourism_project/data/tourism.csv"

OUTPUT_FOLDER = "tourism_project/model_building/data"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

print("="*60)
print("Loading Dataset...")
print("="*60)

df = pd.read_csv(DATA_PATH)

print(f"Dataset Loaded Successfully")
print(f"Shape : {df.shape}")


# ----------------------------------------------------------
# Remove Unnecessary Columns
# ----------------------------------------------------------

columns_to_drop = [

    "CustomerID"

]

df.drop(columns=columns_to_drop,
        inplace=True,
        errors="ignore")

print("\nDropped Columns :", columns_to_drop)

print("New Shape :", df.shape)


# ----------------------------------------------------------
# Train Test Split
# ----------------------------------------------------------

train_df, test_df = train_test_split(

    df,

    test_size=0.20,

    random_state=42,

    stratify=df["ProdTaken"]

)


# ----------------------------------------------------------
# Save Train/Test Dataset
# ----------------------------------------------------------

train_path = os.path.join(OUTPUT_FOLDER,"train.csv")

test_path = os.path.join(OUTPUT_FOLDER,"test.csv")

train_df.to_csv(train_path,index=False)

test_df.to_csv(test_path,index=False)


print("\nTrain Dataset Saved :",train_path)

print("Test Dataset Saved  :",test_path)


print("\nTrain Shape :",train_df.shape)

print("Test Shape  :",test_df.shape)


print("\nData Preparation Completed Successfully.")
