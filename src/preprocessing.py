import numpy as np
import pandas as pd

def Log_Transform(df : pd.DataFrame, columns):
    df = df.copy()

    for column in columns:
        df[column] = np.log1p(df[column])

    return df

def show_outliers_count_IQR(df : pd.DataFrame, columns):
    df = df.copy()
    
    for col in columns:
        Q1 = df[col].quantile(.25)
        Q3 = df[col].quantile(.75)

        IQR = Q3 - Q1

        lower = Q1 -  (1.5 * IQR)
        upper = Q3 + (1.5 * IQR)

        column_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        print(f"{col}  : {column_outliers}")

def remove_outliers_IQR(df : pd.DataFrame, columns):
    df = df.copy()

    for col in columns:
        Q1 = df[col].quantile(.25)
        Q3 = df[col].quantile(.75)

        IQR = Q3 - Q1

        lower = Q1 -  (1.5 * IQR)
        upper = Q3 + (1.5 * IQR)

        df = df[(df[col] > lower) & (df[col] < upper)]

    return df