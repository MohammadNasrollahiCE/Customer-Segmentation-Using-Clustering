import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def save_data(df : pd.DataFrame, file_name):
    df.to_csv(path_or_buf= f'../data/{file_name}.csv')