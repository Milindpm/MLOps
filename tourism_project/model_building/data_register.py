
import pandas as pd
import os

# -------------------------------------------------------
# Dataset Registration Script
# -------------------------------------------------------

DATA_PATH = "tourism_project/data/tourism.csv"

EXPECTED_COLUMNS = [
    'CustomerID',
    'ProdTaken',
    'Age',
    'TypeofContact',
    'CityTier',
    'Occupation',
    'Gender',
    'NumberOfPersonVisiting',
    'PreferredPropertyStar',
    'MaritalStatus',
    'NumberOfTrips',
    'Passport',
    'OwnCar',
    'NumberOfChildrenVisiting',
    'Designation',
    'MonthlyIncome',
    'PitchSatisfactionScore',
    'ProductPitched',
    'NumberOfFollowups',
    'DurationOfPitch'
]

def register_dataset():

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    missing_columns = list(set(EXPECTED_COLUMNS) - set(df.columns))

    if missing_columns:
        raise ValueError(
            f"Dataset validation failed. Missing columns: {missing_columns}"
        )

    print("=" * 60)
    print("DATASET REGISTERED SUCCESSFULLY")
    print("=" * 60)
    print(f"Rows             : {df.shape[0]}")
    print(f"Columns          : {df.shape[1]}")
    print(f"Duplicates       : {df.duplicated().sum()}")
    print(f"Missing Values   : {df.isnull().sum().sum()}")
    print("=" * 60)

    return df


if __name__ == "__main__":
    register_dataset()
