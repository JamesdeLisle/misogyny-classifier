from pandas import DataFrame
from pandas import concat

def prepare_data_set(df: DataFrame):
    df = df[df['is_misogyny'].notna()]
    mis_df = df[df['is_misogyny'] == 1]
    non_df = df[df['is_misogyny'] == 0]
    mis_len = len(mis_df)
    non_len = len(non_df)
    min_len = min([mis_len, non_len])
    clean_df = concat([mis_df.sample(n=min_len), non_df.sample(n=min_len)])
    return clean_df.reset_index(drop=True)