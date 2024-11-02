import pandas as pd
import numpy as np

from utils import (
    calculate_team_results,
    team_confrontations,
    won_games,
    lost_games
)


def win_lost_index(row, df_conf_dict):
    """
    Calculates win and loss punctuation between two teams for the row's season,
    using a precomputed dictionary of relevant confrontations for each season.

    This function returns a victory and lose punctuation between 2 teams.
    It is given by 1 / ((currentseason - season)^2 + 1)
    This function has a maximum in currentseason and it deacreses as season increase
    Giving more importance to recent matches.
    It is symmetric for both teams, so win_team1 = lost_team2.

    :param row: A row from the DataFrame containing "season", "home_team" and "away_team".
    :param df_conf_dict: A dictionary containing relevant confrontations for each season.
    :return: A tuple with the win and loss punctuation for the row's season.
    """
    season = row["season"]
    team1 = row["home_team"]
    team2 = row["away_team"]

    df_conf = df_conf_dict.get((team1, team2), pd.DataFrame())

    df_won = won_games(df_conf, team1)
    if not df_won.empty:
        df_won["win_punct"] = 1 / ((season - df_won["season"].astype(int)) ** 2 + 1)
        win_punct = df_won["win_punct"].sum()
    else:
        win_punct = 0

    df_lost = lost_games(df_conf, team1)
    if not df_lost.empty:
        df_lost["lost_punct"] = 1 / ((season - df_lost["season"].astype(int)) ** 2 + 1)
        lost_punct = df_lost["lost_punct"].sum()
    else:
        lost_punct = 0

    return win_punct, lost_punct


def inform_win_lost_index(df, df_calculate):
    """
    Adds win and loss punctuation to the DataFrame for each row's season.

    :param df: DataFrame containing match data for all teams.
    :param df_calculate: DataFrame containing the matches to calculate the win and loss punctuation.
    :return: DataFrame with the win and loss punctuation for each row's season.
    """
    max_season = df_calculate["season"].max()
    df_recent = df[df["season"] >= (max_season - 20)].copy()

    teams = df_calculate[["home_team", "away_team"]].drop_duplicates()
    df_conf_dict = {
        (team1, team2): team_confrontations(df_recent, team1, team2)
        for team1, team2 in zip(teams["home_team"], teams["away_team"])
    }

    df_calculate[["win_punct", "lost_punct"]] = df_calculate.apply(
        lambda row: win_lost_index(row, df_conf_dict), axis=1, result_type="expand"
    )

    return df_calculate


def difference_points(row, df_results_dict):
    """
    Calculates the difference in points between home and away teams
    considering only matches before the current matchday.

    :param row: DataFrame row with home_team, away_team, season, and matchday
    :param df_results_dict: Dictionary of {season: {matchday: results_df}}
    :return: Point difference between home and away team
    """
    season = row["season"]
    matchday = row["matchday"]

    results_key = (season, matchday)
    if results_key not in df_results_dict:
        return 0

    results_df = df_results_dict[results_key]

    try:
        home_team_points = results_df.loc[
            results_df["team"] == row["home_team"], "points"
        ].iloc[0]
        away_team_points = results_df.loc[
            results_df["team"] == row["away_team"], "points"
        ].iloc[0]
        return home_team_points - away_team_points
    except (KeyError, IndexError):
        return 0


def inform_relatives_points(df, df_calculate):
    """
    Calculates relative points for each match considering only previous matchdays.

    :param df: Complete DataFrame with all matches
    :param df_calculate: DataFrame with matches to calculate features for
    :return: DataFrame with added relative points features
    """
    df_results_dict = {}

    for season in df_calculate["season"].unique():
        season_data = df[df["season"] == season].copy()

        for matchday in df_calculate[df_calculate["season"] == season][
            "matchday"
        ].unique():
            results = calculate_team_results(season_data, max_matchday=matchday)
            df_results_dict[(season, matchday)] = results

    df_calculate["points_relative"] = df_calculate.apply(
        lambda row: difference_points(row, df_results_dict), axis=1
    )

    df_calculate["points_relative_index"] = df_calculate.apply(
        lambda row: row["points_relative"]
        * (row["matchday"] ** 2 / (row["matchday"] ** 2 + 38))
        if row["matchday"] > 0
        else 0,
        axis=1,
    )

    return df_calculate

def calculate_results(df):
    """
    Precalculates results for all matches.

    :param df: DataFrame with match data
    :return: DataFrame with results for all matches
    """
    home_results = df[["matchday", "season", "home_team", "home_win", "tie"]].copy()
    home_results["team"] = home_results["home_team"]
    home_results["Result"] = np.where(
        home_results["tie"] == 1, "T", np.where(home_results["home_win"] == 1, "W", "L")
    )

    away_results = df[["matchday", "season", "away_team", "away_win", "tie"]].copy()
    away_results["team"] = away_results["away_team"]
    away_results["Result"] = np.where(
        away_results["tie"] == 1, "T", np.where(away_results["away_win"] == 1, "W", "L")
    )

    all_results = pd.concat(
        [
            home_results[["matchday", "season", "team", "Result"]],
            away_results[["matchday", "season", "team", "Result"]],
        ]
    ).sort_values(["team", "season", "matchday"], ascending=[True, True, False])

    return all_results


def get_last_5_results(all_results, matchday, team, season):
    """
    Gets all available results up to 5 for a team in a season.

    :param all_results: DataFrame with all results
    :param matchday: Current matchday
    :param team: Team name
    :param season: Season
    :return: Tuple with the last 5 results and the number of matches
    """
    mask = (
        (all_results["team"] == team)
        & (all_results["season"] == season)
        & (all_results["matchday"] < matchday)
    )

    results = all_results[mask]["Result"].head(5).tolist()
    return results, len(results)


def convert_results_to_points(results_tuple):
    """
    Converts results to points with correction factor for fewer than 5 matches.
    The correction factor is 5 / num_matches.

    :param results_tuple: Tuple with results and number of matches
    :return: Corrected points
    """
    results, num_matches = results_tuple
    if num_matches == 0:
        return 0

    points_map = {"W": 3, "L": 0, "T": 1}
    total_points = sum(points_map[result] for result in results)

    correction_factor = 5 / num_matches
    corrected_points = total_points * correction_factor

    return round(corrected_points, 2)


def last5index(df, df_predict):
    """
    Calculates the last 5 matches index for all teams with correction for fewer matches.

    :param df: DataFrame with all matches
    :param df_predict: DataFrame with matches to calculate features for
    :return: DataFrame with added last 5 matches index features
    """
    # df_calculate has the last 5 matches results so takes the same season as df_predict and the matchday is less than the matchday of the match to predict
    df_calculate = df.loc[
        (df["season"] == df_predict["season"].max())
        & (df["matchday"] < df_predict["matchday"].max())
        & (df["matchday"] >= df_predict["matchday"].max() - 5)
    ].copy()

    all_results = calculate_results(df_calculate)

    df_predict["last5_home"] = df_predict.apply(
        lambda row: get_last_5_results(
            all_results, row["matchday"], row["home_team"], row["season"]
        ),
        axis=1,
    )

    df_predict["last5_away"] = df_predict.apply(
        lambda row: get_last_5_results(
            all_results, row["matchday"], row["away_team"], row["season"]
        ),
        axis=1,
    )
    df_predict["last5_home"] = df_predict["last5_home"].apply(convert_results_to_points)
    df_predict["last5_away"] = df_predict["last5_away"].apply(convert_results_to_points)

    return df_predict


def last_season_position(df, df_predict):
    """
    Adds the last season points in a new column.
    If a team was not in the last season, the average of the last season minus 20 is used.

    :param df: DataFrame with all matches
    :param df_predict: DataFrame with matches to calculate features for
    :return: DataFrame with added last season points
    """
    df_calculate = df.loc[(df["season"] == df_predict["season"].max() - 1)].copy()
    df_results = calculate_team_results(df_calculate)

    df_predict["last_season_points_home"] = df_predict.apply(
        lambda row: df_results.loc[
            df_results["team"] == row["home_team"], "points"
        ].iloc[0]
        if row["home_team"] in df_results["team"].values
        else 0,
        axis=1,
    )

    df_predict["last_season_points_away"] = df_predict.apply(
        lambda row: df_results.loc[
            df_results["team"] == row["away_team"], "points"
        ].iloc[0]
        if row["away_team"] in df_results["team"].values
        else 0,
        axis=1,
    )

    # Improvement, calcualte mean of the teams that ascend to the league
    df_predict["last_season_points_home"] = df_predict[
        "last_season_points_home"
    ].replace(0, df_results["points"].mean() - 20)
    df_predict["last_season_points_away"] = df_predict[
        "last_season_points_away"
    ].replace(0, df_results["points"].mean() - 20)
    return df_predict