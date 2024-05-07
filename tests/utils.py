import pandas as pd


def check_df_filled(df: pd.DataFrame) -> bool:
    """
    Check if the dataframe is filled
    :param df: the dataframe
    :return: True if the dataframe is filled, False otherwise
    """
    return df.isnull().sum().sum() == 0


def generate_random_missing_data(df: pd.DataFrame, missing_ratio: float) -> pd.DataFrame:
    """
    Generate random missing data in the dataframe df
    :param df: the dataframe
    :param missing_ratio: the ratio of missing data
    :return: the dataframe with missing data
    """
    df_with_null = df.copy()
    for col in df.columns:
        idx = df.sample(frac=missing_ratio).index
        df_with_null.loc[idx, col] = None
    return df_with_null
