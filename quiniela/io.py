import sqlite3

import pandas as pd
import numpy as np

import settings

def load_matchday(season, division, matchday):
    with sqlite3.connect(settings.DATABASE_PATH) as conn:
        data = pd.read_sql(f"""
            SELECT * FROM Matches
                WHERE season = '{season}'
                  AND division = {division}
                  AND matchday = {matchday}
        """, conn)
        full_data = pd.read_sql("SELECT * FROM Matches", conn)
    if data.empty:
        raise ValueError("There is no matchday data for the values given")
    #define teams_dict
    teams_dict = {}
    teams = np.unique(full_data['home_team'])
    for i in range(len(teams)):
        teams_dict[teams[i]] = i
    #we modify data in order to get the predictors used for our model
    data.dropna(inplace = True)
    data["home_score"] = data.apply(lambda x: int(x["score"].split(":")[0]), axis = 1)
    data["away_score"] = data.apply(lambda x: int(x["score"].split(":")[1]), axis = 1)
    data["winner"] = data.apply(lambda x : x["home_team"] if(x["home_score"] > x["away_score"]) else (x["away_team"] if(x["home_score"] < x["away_score"]) else "NaN"), axis = 1)
    data["loser"] = data.apply(lambda x : x["home_team"] if(x["home_score"] < x["away_score"]) else (x["away_team"] if(x["home_score"] > x["away_score"]) else "NaN"), axis = 1)
    data_ht = data.copy()
    data_ht['team'] = data['home_team']
    data_aw = data.copy()
    data_aw['team'] = data['away_team']
    data_total = pd.concat([data_ht,data_aw])
    data_total = data_total.sort_values(by = ['season', 'division', 'matchday', 'score'])
    data_total['W'] = data_total.apply(lambda x : 1 if x['winner'] == x['team'] else 0, axis = 1)
    data_total['T'] = data_total.apply(lambda x : 1 if x['loser'] == 'NaN' else 0, axis = 1)
    data_total['W'] = data_total.groupby(['season', 'division', 'team'])['W'].cumsum()
    data_total['T'] = data_total.groupby(['season', 'division', 'team'])['T'].cumsum()
    data_total['Pts'] = 3 * data_total['W'] + data_total['T']
    data_total['rank'] = data_total.groupby(['division','season','matchday'])['Pts'].rank(method = 'min', ascending=False)
    data_total = data_total.sort_index()
    data_total_ht = data_total.loc[data_total['home_team'] == data_total['team']]
    data_total_aw = data_total.loc[data_total['away_team'] == data_total['team']]
    data['home_team_rank'] = data_total_ht['rank']
    data['away_team_rank'] = data_total_aw['rank']
    data.drop(columns = ['winner', 'loser'], inplace = True)
    data.drop(columns = ['home_score', 'away_score'], inplace = True) 
    # now we define the column to predict and assign to each team a number in order to use the teams as predictors
    data['winner'] = data['score'].str.split(':').str[0].astype(int) - data['score'].str.split(':').str[1].astype(int)
    data['winner'] = np.where(data['winner'] > 0, 0, np.where(data['winner'] < 0, 2, 1))
    teams = [data['home_team'].unique()]
    teams = teams[0].tolist()
    print(teams_dict['Barcelona'])
    data['home_team_num'] = data['home_team'].map(teams_dict)
    data['away_team_num'] = data['away_team'].map(teams_dict)
    data['winner'] = data['winner'].astype(int)
    data['home_team_num'] = data['home_team_num'].astype(int)
    data['away_team_num'] = data['away_team_num'].astype(int)
    return data


def load_historical_data(seasons):
    with sqlite3.connect(settings.DATABASE_PATH) as conn:
        if seasons == "all":
            data = pd.read_sql("SELECT * FROM Matches", conn)
        else:
            data = pd.read_sql(f"""
                SELECT * FROM Matches
                    WHERE season IN {tuple(seasons)}
            """, conn)
        full_data = pd.read_sql("SELECT * FROM Matches", conn)
    if data.empty:
        raise ValueError(f"No data for seasons {seasons}")
    #define teams_dict
    teams_dict = {}
    teams = np.unique(full_data['home_team'])
    for i in range(len(teams)):
        teams_dict[teams[i]] = i
    #we modify data in order to get the predictors used for our model
    data.dropna(inplace = True)
    data["home_score"] = data.apply(lambda x: int(x["score"].split(":")[0]), axis = 1)
    data["away_score"] = data.apply(lambda x: int(x["score"].split(":")[1]), axis = 1)
    data["winner"] = data.apply(lambda x : x["home_team"] if(x["home_score"] > x["away_score"]) else (x["away_team"] if(x["home_score"] < x["away_score"]) else "NaN"), axis = 1)
    data["loser"] = data.apply(lambda x : x["home_team"] if(x["home_score"] < x["away_score"]) else (x["away_team"] if(x["home_score"] > x["away_score"]) else "NaN"), axis = 1)
    data_ht = data.copy()
    data_ht['team'] = data['home_team']
    data_aw = data.copy()
    data_aw['team'] = data['away_team']
    data_total = pd.concat([data_ht,data_aw])
    data_total = data_total.sort_values(by = ['season', 'division', 'matchday', 'score'])
    data_total['W'] = data_total.apply(lambda x : 1 if x['winner'] == x['team'] else 0, axis = 1)
    data_total['T'] = data_total.apply(lambda x : 1 if x['loser'] == 'NaN' else 0, axis = 1)
    data_total['W'] = data_total.groupby(['season', 'division', 'team'])['W'].cumsum()
    data_total['T'] = data_total.groupby(['season', 'division', 'team'])['T'].cumsum()
    data_total['Pts'] = 3 * data_total['W'] + data_total['T']
    data_total['rank'] = data_total.groupby(['division','season','matchday'])['Pts'].rank(method = 'min', ascending=False)
    data_total = data_total.sort_index()
    data_total_ht = data_total.loc[data_total['home_team'] == data_total['team']]
    data_total_aw = data_total.loc[data_total['away_team'] == data_total['team']]
    data['home_team_rank'] = data_total_ht['rank']
    data['away_team_rank'] = data_total_aw['rank']
    data.drop(columns = ['winner', 'loser'], inplace = True)
    data.drop(columns = ['home_score', 'away_score'], inplace = True) 
    # now we define the column to predict and assign to each team a number in order to use the teams as predictors
    data['winner'] = data['score'].str.split(':').str[0].astype(int) - data['score'].str.split(':').str[1].astype(int)
    data['winner'] = np.where(data['winner'] > 0, 0, np.where(data['winner'] < 0, 2, 1))
    data['home_team_num'] = data['home_team'].map(teams_dict)
    data['away_team_num'] = data['away_team'].map(teams_dict)
    data['winner'] = data['winner'].astype(int)
    data['home_team_num'] = data['home_team_num'].astype(int)
    data['away_team_num'] = data['away_team_num'].astype(int)
    return data


def save_predictions(predictions):
    predictions.drop(columns = ['home_team_rank', 'away_team_rank', 'home_team_num', 'away_team_num'], inplace = True)
    predictions.rename(columns = {'winner' : 'pred'}, inplace = True)
    with sqlite3.connect(settings.DATABASE_PATH) as conn:
        predictions.to_sql(name="Predictions", con=conn, if_exists="append", index=False)
