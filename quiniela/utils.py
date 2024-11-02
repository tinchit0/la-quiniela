def calculate_team_results(df, max_matchday=None):
    """
    Calculates team results up to a specific matchday.
    Has been modified to include just the necessary columns for the model.

    :param df: DataFrame with match data
    :param max_matchday: Only consider matches up to this matchday (exclusive)
    :return: DataFrame with team points
    """
    if max_matchday is not None:
        df = df[df["matchday"] < max_matchday].copy()

    df_results = (
        pd.concat(
            [
                df.groupby(["season", "home_team"])
                .agg(
                    W=("home_win", "sum"),
                    L=("away_win", "sum"),
                    T=("tie", "sum"),
                )
                .reset_index()
                .rename(columns={"home_team": "team"}),
                df.groupby(["season", "away_team"])
                .agg(
                    W=("away_win", "sum"),
                    L=("home_win", "sum"),
                    T=("tie", "sum"),
                )
                .reset_index()
                .rename(columns={"away_team": "team"}),
            ]
        )
        .groupby(["season", "team"])
        .sum()
        .reset_index()
    )

    df_results["points"] = df_results["W"] * 3 + df_results["T"]

    df_results = df_results.sort_values(
        by=["season", "points"],
        ascending=[False, False],
    ).reset_index(drop=True)

    df_results["rank"] = (
        df_results.groupby(["season"])["points"]
        .rank("first", ascending=False)
        .astype(int)
    )

    df_results = df_results[
        [
            "season",
            "team",
            "points",
        ]
    ]
    return df_results

def team_confrontations(df, team1, team2):
    """
    Returns the confrontations between two teams

    :param df: DataFrame with match data
    :param team1: First team
    :param team2: Second team
    :return: DataFrame with confrontations between the two teams
    """
    df_confrontations = df.loc[
        ((df["home_team"] == team1) | (df["away_team"] == team1))
        & ((df["home_team"] == team2) | (df["away_team"] == team2))
    ]

    return df_confrontations


def won_games(df, team: str):
    """
    Returns those winning games for a given team

    :param df: DataFrame with match data
    :param team: Team name
    :return: DataFrame with winning games for the team
    """
    home_wins = (df["home_team"] == team) & (df["home_win"] == 1)
    away_wins = (df["away_team"] == team) & (df["away_win"] == 1)
    return df[home_wins | away_wins]


def lost_games(df, team: str):
    """
    Returns those winning games for a given team

    :param df: DataFrame with match data
    :param team: Team name
    :return: DataFrame with winning games for the team
    """
    home_lost = (df["home_team"] == team) & (df["home_win"] == 0)
    away_lost = (df["away_team"] == team) & (df["away_win"] == 0)
    tie_games = (df["home_team"] == team) & (df["tie"] == 1) | (
        df["away_team"] == team
    ) & (df["tie"] == 1)

    return df[(home_lost | away_lost) & ~tie_games]