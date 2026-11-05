import pandas as pd


df = pd.read_parquet(r'C:\ship_pj\data\train_01.parquet')
print(df.shape)
print(df.dtypes)
print(df.isna().sum())
print(df['fishing_code'].value_counts())
print(df.groupby('fishing_code')['rfid_id'].nunique())
print(df.describe().T)              