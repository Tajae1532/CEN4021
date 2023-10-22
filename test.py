import pandas as pd
import numpy as np
from sportsipy.nfl.boxscore import Boxscores, Boxscore
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', None)

#Cairo commits

# Define a date range
start_date = '2022-09-08'
end_date = '2022-10-12'

def get_elo():
    elo_df = pd.read_csv(r"C:\Users\ander\OneDrive\Desktop\nfl_elo_latest.csv")
    elo_df = elo_df.drop(columns = ['season','neutral' ,'playoff', 'elo_prob1', 'elo_prob2', 'elo1_post', 'elo2_post',
           'qbelo1_pre', 'qbelo2_pre', 'qb1', 'qb2', 'qb1_adj', 'qb2_adj', 'qbelo_prob1', 'qbelo_prob2',
           'qb1_game_value', 'qb2_game_value', 'qb1_value_post', 'qb2_value_post',
           'qbelo1_post', 'qbelo2_post', 'score1', 'score2'])
    elo_df.date = pd.to_datetime(elo_df.date)
    elo_df = elo_df[elo_df.date < '01-05-2022']

    elo_df['team1'] = elo_df['team1'].replace(['KC', 'JAX', 'CAR', 'BAL', 'BUF', 'MIN', 'DET', 'ATL', 'NE', 'WSH',
           'CIN', 'NO', 'SF', 'LAR', 'NYG', 'DEN', 'CLE', 'IND', 'TEN', 'NYJ',
           'TB', 'MIA', 'PIT', 'PHI', 'GB', 'CHI', 'DAL', 'ARI', 'LAC', 'HOU',
           'SEA', 'OAK'],
            ['kan','jax','car', 'rav', 'buf', 'min', 'det', 'atl', 'nwe', 'was', 
            'cin', 'nor', 'sfo', 'ram', 'nyg', 'den', 'cle', 'clt', 'oti', 'nyj', 
             'tam','mia', 'pit', 'phi', 'gnb', 'chi', 'dal', 'crd', 'sdg', 'htx', 'sea', 'rai' ])
    elo_df['team2'] = elo_df['team2'].replace(['KC', 'JAX', 'CAR', 'BAL', 'BUF', 'MIN', 'DET', 'ATL', 'NE', 'WSH',
           'CIN', 'NO', 'SF', 'LAR', 'NYG', 'DEN', 'CLE', 'IND', 'TEN', 'NYJ',
           'TB', 'MIA', 'PIT', 'PHI', 'GB', 'CHI', 'DAL', 'ARI', 'LAC', 'HOU',
           'SEA', 'OAK'],
            ['kan','jax','car', 'rav', 'buf', 'min', 'det', 'atl', 'nwe', 'was', 
            'cin', 'nor', 'sfo', 'ram', 'nyg', 'den', 'cle', 'clt', 'oti', 'nyj', 
             'tam','mia', 'pit', 'phi', 'gnb', 'chi', 'dal', 'crd', 'sdg', 'htx', 'sea', 'rai' ])
    return elo_df


def get_schedule(start_date, end_date):
    schedule_df = pd.DataFrame()
    date_range = pd.date_range(start=start_date, end=end_date)
    
    for date in date_range:
        date_string = date.strftime('%Y-%m-%d')
        week_scores = Boxscores(date_string)
        week_games_df = pd.DataFrame()
        
        for _, games in week_scores.games.items():
            for game in games:
                game_df = pd.DataFrame(game, index=[0])[
                    ['away_name', 'away_abbr', 'home_name', 'home_abbr', 'winning_name', 'winning_abbr']
                ]
                game_df['date'] = date
                week_games_df = pd.concat([week_games_df, game_df])
        
        schedule_df = pd.concat([schedule_df, week_games_df]).reset_index(drop=True)
    
    return schedule_df

def merge_rankings(agg_games_df,elo_df):
    agg_games_df = pd.merge(agg_games_df, elo_df, how = 'inner', left_on = ['home_abbr', 'away_abbr'], right_on = ['team1', 'team2']).drop(columns = ['date','team1', 'team2'])
    agg_games_df['elo_dif'] = agg_games_df['elo2_pre'] - agg_games_df['elo1_pre']
    agg_games_df['qb_dif'] = agg_games_df['qb2_value_pre'] - agg_games_df['qb1_value_pre']
    agg_games_df = agg_games_df.drop(columns = ['elo1_pre', 'elo2_pre', 'qb1_value_pre', 'qb2_value_pre'])
    return agg_games_df

def prep_test_train(current_week,weeks,year):
    schedule_df = get_schedule(start_date, end_date)
    # weeks_games_df = game_data_up_to_week(weeks,year)
    agg_games_df = merge_rankings(schedule_df, elo_df)
    elo_df = get_elo(start_date, end_date)
    agg_games_df = merge_rankings(agg_games_df, elo_df)
    train_df = agg_games_df[agg_games_df.result.notna()]
    return test_df, train_df

current_week = '2023-02-12'  # Specify the current date
year = 2022
test_df, train_df = prep_test_train(start_date, current_week, year)
