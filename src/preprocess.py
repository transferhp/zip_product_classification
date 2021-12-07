from pathlib import Path
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from utils import load_raw_data


def data_processing():
    base_path = Path(__file__).parent.parent / "data"
    print(f"Logging Info - Load raw data from {base_path}")
    data = load_raw_data(file_path=f"{base_path}/exercise3.jl.zip")
    # Create DataFrame from dict
    df = pd.DataFrame.from_dict(data)
    # Combine level0, level1 and level2 as a flat category
    df["cat0_cat1_cat2"] = df["cat0"] + "|" + df["cat1"] + "|" + df["cat2"]
    # Split into train/test data
    train_df, test_df = train_test_split(shuffle(df), test_size=0.2, random_state=123)
    # Save results
    df.to_csv(f"{base_path}/data.csv")
    train_df.to_csv(f"{base_path}/train.csv")
    test_df.to_csv(f"{base_path}/test.csv")
