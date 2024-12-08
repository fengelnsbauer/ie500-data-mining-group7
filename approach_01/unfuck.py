import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def n_to_nan(x):
    if x == "\\N":
        return np.nan
    return x


def race_time_to_milliseconds(race_time_str):
    """
    Converts a Formula 1 race time string (e.g., "1:20.8888") to milliseconds.

    Parameters:
        race_time_str (str): Race time as a string in the format "M:SS.ssss".

    Returns:
        float: Race time in milliseconds.
    """
    if not isinstance(race_time_str, str):
        return np.nan
    try:
        # Split the string into minutes and seconds
        minutes, seconds = race_time_str.split(":")

        # Convert minutes to milliseconds
        minutes_ms = int(minutes) * 60 * 1000

        # Convert seconds (with fractional part) to milliseconds
        seconds_ms = float(seconds) * 1000

        # Total milliseconds
        total_ms = minutes_ms + seconds_ms

        return total_ms
    except Exception as e:
        raise ValueError(f"Invalid race time format: {race_time_str}") from e


def unfuck_data(df):

    df = df.copy()
    # Convert race_date and driver_dob to datetime
    df["race_date"] = pd.to_datetime(df["race_date"])
    df["driver_dob"] = pd.to_datetime(df["driver_dob"])

    # Calculate age in milliseconds
    df["driver_age"] = (df["race_date"] - df["driver_dob"]).dt.total_seconds() * 1000
    df["race_date"] = df["race_date"].astype("int64") // 10**6

    df.drop(
        columns=[
            "resultId",
            "raceId",
            "number",
            "q1_time",
            "q2_time",
            "q3_time",
            "fp1_date",
            "fp1_time",
            "fp2_date",
            "fp2_time",
            "fp3_date",
            "fp3_time",
            "race_time",
            "race_name",
            "quali_time",
            "quali_date",
            "driver_dob",
        ],
        inplace=True,
    )
    df = df.apply(lambda x: x.apply(n_to_nan))

    df["q1"] = df["q1"].apply(race_time_to_milliseconds)
    df["q2"] = df["q2"].apply(race_time_to_milliseconds)
    df["q3"] = df["q3"].apply(race_time_to_milliseconds)

    # df["q1"].fillna(0, inplace=True)
    # df["q2"].fillna(0, inplace=True)
    # df["q3"].fillna(0, inplace=True)

    df["q2"] = df["q2"].fillna(df["q1"])
    df["q3"] = df["q3"].fillna(df["q2"])

    labelencoder = LabelEncoder()

    for column in df.columns:
        if df[column].dtype == type(object):
            df[column] = labelencoder.fit_transform(df[column])

    return df
