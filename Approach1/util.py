import pandas as pd


def time_delta_to_millis(time_delta) -> float:
    return time_delta.total_seconds().astype(float) * 1000


def convert_timedelta_columns_to_millis(df: pd.DataFrame, columns: list, in_place: bool = False) -> pd.DataFrame:
    """
    Convert specified timedelta columns in a DataFrame to milliseconds.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the timedelta columns.
    columns (list): List of column names to convert.
    inPlace (bool): If True, modify the DataFrame in place. Default is False.

    Returns:
    pd.DataFrame: The DataFrame with specified columns converted to milliseconds.
    """
    if not in_place:
        df = df.copy()
    for col in columns:
        df[col] = (df[col].dt.total_seconds() * 1000).astype(float)
    return df

def generate_empty_session_df(session_name: str, race_id) -> pd.DataFrame:
    """
    Generate an empty DataFrame with columns for the specified session type.

    Parameters:
    session (str): The session type (e.g. 'fp1', 'fp2', 'fp3')

    Returns:
    pd.DataFrame: An empty DataFrame with columns for the specified session type.
    """
    return pd.DataFrame(columns=[
          'raceId', 'driver_code',
          f'{session_name}_avg_sector_1', f'{session_name}_avg_sector_2', f'{session_name}_avg_lap_time',
          f'{session_name}_avg_speedI1', f'{session_name}_avg_speedI2', f'{session_name}_avg_speedFL',
          f'{session_name}_avg_speedST', f'{session_name}_avg_tyre_life', f'{session_name}_avg_is_on_fresh_tyres'
      ], data={
          'raceId': race_id,
          'driver_code': [],
          f'{session_name}_avg_sector_1': [],
          f'{session_name}_avg_sector_2': [],
          f'{session_name}_avg_lap_time': [],
          f'{session_name}_avg_speedI1': [],
          f'{session_name}_avg_speedI2': [],
          f'{session_name}_avg_speedFL': [],
          f'{session_name}_avg_speedST': [],
          f'{session_name}_avg_tyre_life': [],
          f'{session_name}_avg_is_on_fresh_tyres': []
          })