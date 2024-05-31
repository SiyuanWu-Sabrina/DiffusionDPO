import pandas as pd
from config import *


def parquet_to_csv(parquet_path: str, csv_path: str):
    print("Converting parquet to csv...")
    data = pd.read_parquet(parquet_path)
    data.to_csv(csv_path, index=False)


if __name__ == "__main__":
    # parquet_to_csv(Stable_Diffusion_Parquet_Train_Path, Stable_Diffusion_CSV_Train_Path)
    parquet_to_csv(Stable_Diffusion_Parquet_Val_Path, Stable_Diffusion_CSV_Val_Path)