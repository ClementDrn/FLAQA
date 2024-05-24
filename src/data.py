import os
import pandas as pd;
import numpy as np;
from scipy import stats; 


# --- Configuration starts here ----------------------------------------------

# Paths to the CSV files containing the data
EXT_DATA_PATH = "../data/Panel Title-data-2024-04-09 23 51 35__EXT.csv"
IN1_DATA_PATH = "../data/Panel Title-data-2024-04-09 23 51 56__IN1.csv"
IN2_DATA_PATH = "../data/Panel Title-data-2024-04-09 23 52 12__IN2.csv"

# --- Configuration ends here ------------------------------------------------


def load_data(range_start: pd.Timestamp, range_end: pd.Timestamp) -> dict:
    # Read CSV data 
    dir_path = os.path.dirname(os.path.realpath(__file__))
    df_ext = pd.read_csv(os.path.join(dir_path, EXT_DATA_PATH), index_col=0)
    df_in1 = pd.read_csv(os.path.join(dir_path, IN1_DATA_PATH), index_col=0)
    df_in2 = pd.read_csv(os.path.join(dir_path, IN2_DATA_PATH), index_col=0)
    dataframes = { 
        "EXT": df_ext, 
        "IN1": df_in1, 
        "IN2": df_in2
    }

    # Convert table indices to datetime type
    for df in dataframes.values():
        df.index = pd.to_datetime(df.index)

    # If range start and end are unspecified, 
    # they default to the minimum and maximum index between all dataframes
    if range_start is None:
        range_start = max([df.index[0] for df in dataframes.values()])
    if range_end is None:
        range_end = min([df.index[-1] for df in dataframes.values()])

    # Cut out data that is out of range
    for df in dataframes.values():
        df = df[(df.index >= range_start) & (df.index <= range_end)]

    # Round indices to the nearest minute
    for df in dataframes.values():
        df.index = df.index.round("min")

    # Make sure that start and end range indices exist dataframes
    for df in dataframes.values():
        if range_start < df.index[0]:
            df.loc[range_start] = np.nan
        if range_end > df.index[-1]:
            df.loc[range_end] = np.nan

    # Check for missing data
    # There should be data at each minute.
    for df in dataframes.values():
        i = 1
        last_timestamp = df.index[0]
        # If data is missing, fill it with NA
        while i < len(df.index):
            if df.index[i] - last_timestamp > pd.Timedelta(minutes=1):
                df.loc[last_timestamp + pd.Timedelta(minutes=1)] = np.nan
            last_timestamp = df.index[i]
            i += 1

    # Sort dataframes by index
    for df in dataframes.values():
        df.sort_index(inplace=True)
        
    return dataframes


# Filter dataframe and remove outliers from provided dataframe
# Returns the outliers that were removed
def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    # Dataframe to store outliers
    outliers = pd.DataFrame(index=df.index, columns=df.columns)

    # Filter out values outside of 3 standard deviations
    # These values are replaced with NaN and are saved in the outliers dataframe
    for col in df.columns:
        # Calculate z-scores with NaN values filled with the mean
        z_scores = stats.zscore(df[col].fillna(df[col].mean()))
        # Identify outliers where the value is not NaN
        outlier_mask = (np.abs(z_scores) > 3) & (df[col].notna())
        # Save outliers
        outliers.loc[outlier_mask, col] = df.loc[outlier_mask, col]
        # Replace outliers with NaN
        df.loc[outlier_mask, col] = np.nan

    return outliers


def remove_gradient_outliers(df: pd.DataFrame, col) -> pd.DataFrame:
    # Dataframe to store outliers
    outliers = pd.DataFrame(index=df.index, columns=df.columns)

    # Filter out values outside of 3 standard deviations
    # These values are replaced with NaN and are saved in the outliers dataframe

    # Calculate gradient
    gradient = df[col].diff()
    # Calculate z-scores with NaN values filled with the mean
    z_scores = stats.zscore(gradient.fillna(gradient.mean()))
    # Identify outliers where the value is not NaN
    outlier_mask = (np.abs(z_scores) > 3) & (df[col].notna())
    # Save outliers
    outliers.loc[outlier_mask, col] = df.loc[outlier_mask, col]
    # Replace outliers with NaN
    df.loc[outlier_mask, col] = np.nan

    return outliers
