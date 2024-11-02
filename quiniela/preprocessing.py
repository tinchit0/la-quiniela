import pandas as pd

def _process_scores(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the score column and calculate derived features.
    
    :param df: Input DataFrame with score column
    :return: DataFrame with processed score columns
    """
    df = df.dropna(subset=["score"])
    score_parts = df["score"].str.split(":", expand=True)
    df["home_score"] = pd.to_numeric(score_parts[0], errors="coerce")
    df["away_score"] = pd.to_numeric(score_parts[1], errors="coerce")
    df["difference_score"] = abs(df["home_score"] - df["away_score"])
    return df

def _calculate_match_results(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate match results and create result indicators.
    
    :param df: Input DataFrame with processed scores
    :return: DataFrame with added result columns
    """
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df["away_win"] = (df["home_score"] < df["away_score"]).astype(int)
    df["tie"] = (df["home_score"] == df["away_score"]).astype(int)

    # Create final result column (-1: away win, 0: tie, 1: home win)
    conditions = [
        (df["home_win"] == 1),
        (df["tie"] == 1),
        (df["away_win"] == 1)
    ]
    choices = [1, 0, -1]
    df["result"] = np.select(conditions, choices, default=None)
    return df

def _normalize_dates(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize dates by appending the correct century based on the season information.
    
    :param df: DataFrame containing 'date' and 'season' columns
    :return: DataFrame with normalized dates
    """
    def adjust_date(row: pd.Series) -> str:
        """Adjust a date entry to include the full year based on the season context."""
        start_year, end_year = map(int, row["season"].split("-"))
        date_components = row["date"].split("/")
        short_year = int(date_components[-1])
        full_year = start_year if short_year == start_year % 100 else end_year
        return f"{date_components[0]}/{date_components[1]}/{full_year}"

    df["date"] = df.apply(adjust_date, axis=1)
    df = df.dropna(subset=["date"])
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    return df

def _process_seasons(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Process season information to extract the starting year.
    
    :param df: Input DataFrame with season column
    :return: DataFrame with processed season column
    """
    df["season"] = df["season"].str.split("-").str[0].astype(int)
    return df