import re
import pandas as pd
from pandas.core.dtypes.common import is_object_dtype


APP_ID_COL = 'SK_ID_CURR'
ID_COLS = [APP_ID_COL, 'Sk_ID_CURR', 'SK_ID_PREV', 'SK_ID_BUREAU']
TARGET_COL = 'TARGET'
ADDITIONAL_CATEGORICAL_COLS = [
        "REG_REGION_NOT_LIVE_REGION",
        "REG_REGION_NOT_WORK_REGION",
        "LIVE_REGION_NOT_WORK_REGION",
        "REG_CITY_NOT_LIVE_CITY",
        "REG_CITY_NOT_WORK_CITY",
        "LIVE_CITY_NOT_WORK_CITY",
        "REGION_RATING_CLIENT",
        "REGION_RATING_CLIENT_W_CITY"
]


def set_categorical_types(df, print_warning=True):
    flag_pattern = re.compile(r"^n?(flag_).*?", flags=re.IGNORECASE)
    columns = df.columns.tolist()
    for c in columns:
        if flag_pattern.search(c) or is_object_dtype(df[c]) or c in ADDITIONAL_CATEGORICAL_COLS:
            df[c] = df[c].astype("category")
        else:
            if print_warning and df[c].nunique() <= 2 and c != TARGET_COL:
                unique_values = df[c].unique()
                print(f"Column {c} has unique values: {unique_values}. Should it be categorical?")

def load_table(table_path,
               columns=None,
               print_warning=True):
    """
    Loads a table from a csv file and sets all columns representing categorical values to the 'category' type.
    :param table_path: The file path for the table.
    :param columns: Optional. The only columns that should be loaded. None for all columns.
    :param print_warning: Warns of columns with up to 2 unique values.
    :return: A dataframe with the loaded table.
    """
    if columns is None:
        df = pd.read_csv(table_path)
    else:
        df = pd.read_csv(table_path, usecols=columns)
    set_categorical_types(df, print_warning)
    return df

def get_numeric_cols(df):
    # All non-categorical columns in this project are numerical.
    return [col for col in df.columns
            if df[col].dtype != 'category' and col not in ID_COLS]

def get_categorical_cols(df):
    return [col for col in df.columns if df[col].dtype == 'category']
