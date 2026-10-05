import pandas as pd

def solution(df):
    df['name'] = df['name'].str.strip().str.title()
    df['date'] = pd.to_datetime(df['date'].str.strip()).dt.strftime('%Y-%m-%d')
    df = df.drop_duplicates().reset_index(drop=True)
    df['value'] = df['value'].fillna(df['value'].mean())

    return df