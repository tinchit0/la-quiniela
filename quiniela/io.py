import sqlite3
import os
import sys

import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import settings


def load_matchday(season, division, matchday):
    with sqlite3.connect(settings.DATABASE_PATH) as conn:
        data = pd.read_sql(f"""
            SELECT * FROM Matches
                WHERE season = '{season}'
                  AND division = {division}
                  AND matchday = {matchday}
        """, conn)
    if data.empty:
        raise ValueError("There is no matchday data for the values given")
    return data


def load_historical_data(season: str, depth: int) -> pd.DataFrame:
    """
    Load historical match data for the specified season and depth.
    
    :param season: The starting season in the format "YYYY/YYYY"
    :param depth: The number of seasons to load, including the specified season
    :return: DataFrame containing match data for the specified seasons
    :raises ValueError: If no data is available for the specified seasons
    """
    def get_season_list(start_season: str, depth: int) -> list:
        """
        Generate a list of seasons based on the starting season and depth.
        
        :param start_season: The starting season in "YYYY/YYYY" format
        :param depth: The number of seasons to include
        :return: List of season strings in "YYYY/YYYY" format
        """
        start_year = int(start_season.split("/")[0])
        return [f"{start_year - i}-{(start_year - i + 1)}" for i in range(depth)]

    seasons = get_season_list(season, depth)

    with sqlite3.connect(settings.DATABASE_PATH) as conn:
        data = pd.read_sql(
            f"SELECT * FROM Matches WHERE season IN {tuple(seasons)}", conn
        )

    if data.empty:
        raise ValueError(f"No data for seasons {seasons}")

    return data



def save_predictions(predictions):
    with sqlite3.connect(settings.DATABASE_PATH) as conn:
        predictions.to_sql(name="Predictions", con=conn, if_exists="append", index=False)

