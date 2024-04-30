import os
from warnings import filterwarnings

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from tqdm.auto import tqdm

from jeffutils.utils import movecol

filterwarnings('ignore')
pd.set_option('display.max_columns', None)


def get_team_df(team_one, df_last_two):
    """
    Generates a DataFrame containing features relevant to team_one's performance 
    and interactions with opponents from previous two seasons.

    Args:
    - team_one (str): Name of the team for which the DataFrame is generated.
    - df_last_two (DataFrame): DataFrame containing data for the last two seasons.

    Returns:
    - DataFrame: A DataFrame with the following columns:
        - 'team': Name of the selected team.
        - 'is_home': Binary indicator if the selected team played at home (1) or away (0).
        - 'curr_strength_code': Current strength code of the selected team.
        - 'curr_goals': Number of goals scored by the selected team in the current match.
        - 'curr_opponent_goals': Number of goals scored by the opponent in the current match.
        - 'win': Binary indicator if the selected team won the current match (1) or not (0).
        - 'team_STATE': Features related to the selected team's performance in various states.
        - 'opponent_STATE': Features related to the opponent's performance in various states.
        - 'opp': Name of the opponent.
        - One-hot encoded columns for opponents and current strength codes.
    """
    
    # Get a subset of the data that just has team_one
    df_team_one = df_last_two[(df_last_two['home_name'] == team_one) | (df_last_two['away_name'] == team_one)]

    # Create a new column called 'opponent' that is the name of the team that team_one is playing
    df_team_one['opponent'] = np.where(df_team_one['home_name'] == team_one, df_team_one['away_name'], df_team_one['home_name'])

    # Create a new column if the selected team is the home team or not
    df_team_one['is_home'] = (df_team_one['home_name'] == team_one).astype(int)
    df_team_one['team'] = team_one

    # Get the scored goals for the selected team
    df_team_one['final_goals'] = np.where(df_team_one['home_name'] == team_one, df_team_one['home_final'], df_team_one['away_final'])

    # Get the scored goals for the opponent
    df_team_one['final_opponent_goals'] = np.where(df_team_one['home_name'] == team_one, df_team_one['away_final'], df_team_one['home_final'])

    # Get the current scored goals for the selected team and the opponent
    df_team_one['curr_goals'] = np.where(df_team_one['home_name'] == team_one, df_team_one['home_score'], df_team_one['away_score'])
    df_team_one['curr_opponent_goals'] = np.where(df_team_one['home_name'] == team_one, df_team_one['away_score'], df_team_one['home_score'])

    # Create home and away strength codes
    df_team_one['home_strength_code'] = df_team_one['strength_code']
    df_team_one['away_strength_code'] = 'EV'
    df_team_one.loc[df_team_one['home_strength_code'] == 'SH', 'away_strength_code'] = 'PP'
    df_team_one.loc[df_team_one['home_strength_code'] == 'PP', 'away_strength_code'] = 'SH'

    # Get the current strength code for the selected
    df_team_one['curr_strength_code'] = np.where(df_team_one['home_name'] == team_one, df_team_one['home_strength_code'], df_team_one['away_strength_code'])

    # Get if the selected team won or not
    # df_team_one.drop(columns=['home_final', 'away_final', win], inplace=True)
    df_team_one['win'] = (df_team_one['final_goals'] > df_team_one['final_opponent_goals']).astype(int)
    df_team_one = movecol(df_team_one, ['team', 'opponent', 'is_home', 'curr_strength_code', 'curr_goals', 'curr_opponent_goals', 'win'], 'strength_code', 'Before')

    # Drop a bunch of columns
    df_team_one.drop(columns=['home_score', 'away_score', 'home_final', 'away_final', 'home_name', 'away_name', 'strength_code',
                            'final_goals', 'final_opponent_goals', 'home_strength_code', 'away_strength_code'], inplace=True)

    # Get all columns with 'STATE_'
    state_columns_pre = df_team_one.columns[df_team_one.columns.str.contains('STATE_')]
    state_columns = list(set(["_".join(state.split('_')[:-1]) for state in state_columns_pre]))
    sorted(state_columns)

    # Iterate through each state getting the appropriate value for the selected team.
    for state in state_columns:
        df_team_one[f"team_{state}"] = np.where(df_team_one['is_home'] == 1, df_team_one[f'{state}_HOME'], df_team_one[f'{state}_AWAY'])

    # Iterate through each state getting the appropriate value for the opponent.
    for state in state_columns:
        df_team_one[f"opponent_{state}"] = np.where(df_team_one['is_home'] == 1, df_team_one[f'{state}_AWAY'], df_team_one[f'{state}_HOME'])    

    # Drop state_columns_pre
    df_team_one.drop(columns=state_columns_pre, inplace=True)

    # one hot opponent
    df_team_one['opp'] = df_team_one['opponent']
    return df_team_one


def get_catboost_and_pickle(team_one, df_train, df_test):
    '''
    Trains a CatBoost classifier on data for team_one and saves the model.

    Args:
    - team_one (str): Name of the team for which the model is trained.
    - df_train (DataFrame): The training dataset.
    - df_test (DataFrame): The testing dataset.

    Returns:
    - float: The accuracy of the model on the test dataset.

    This function trains a CatBoost model on the provided training data, performs grid search for hyperparameter tuning, 
    evaluates the model on the test data, selects the best model, and saves it to a file using built-in catboost model saving.
    The test accuracy of the model is returned.
    '''
    
    # Get the data.
    df_team_one_train = get_team_df(team_one, df_train)
    df_team_one_test = get_team_df(team_one, df_test)
    
    # Get X and y
    X_train = df_team_one_train.drop(columns=['game_id', 'team', 'win', 'opp'])
    y_train = df_team_one_train['win']

    X_test = df_team_one_test.drop(columns=['game_id', 'team', 'win', 'opp'])
    y_test = df_team_one_test['win']
    
    # Grid search parameters
    params = {
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 500, 1000],
        'max_depth': [3, 4, 5, 6, 7],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bylevel': [0.6, 0.8, 1.0],
        'cat_features': [['opponent', 'curr_strength_code']],
        'l2_leaf_reg': [1, 3, 5],
        'min_child_samples': [5, 7, 10],
        'random_seed': [42],  # Keep it constant
        'loss_function': ['Logloss'],  # For binary classification
        'eval_metric': ['AUC'],        # Metric for evaluation
        'verbose': [0]
    }
    
    # Define cross-validation strategy
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    
    # Define the model
    model = CatBoostClassifier()
    
    # Perform GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=params, cv=cv, scoring='accuracy', n_jobs=80, refit=True)
    grid_search.fit(X_train, y_train)
    test_accuracy = grid_search.score(X_test, y_test)
    
    # Get the best model
    model = grid_search.best_estimator_

    # Save the model
    model.save_model(f'team_catboost_files/{team_one}')
    
    return test_accuracy



if __name__ == '__main__':
    # Read in the data
    df = pd.read_feather('./data/play_by_play_full_state_space.feather')
    df = df.loc[:, ~df.columns.str.contains('_raw')]

    df['home_name'] = df['home_name'].replace({'Montréal Canadiens': 'Montreal Canadiens',
                                            'MontrÃ©al Canadiens': 'Montreal Canadiens'})
    df['away_name'] = df['away_name'].replace({'Montréal Canadiens': 'Montreal Canadiens',
                                                'MontrÃ©al Canadiens': 'Montreal Canadiens'})

    # Get only the rows with game_date >= 2021-10-01
    df_last_two = df[df['game_date'] >= '2018-10-01']

    df_train, df_test = df_last_two[df_last_two['game_date'] < '2023-10-01'], df_last_two[df_last_two['game_date'] >= '2023-10-01']

    # Drop columns we do not need for XGBoost
    drop_columns = ['game_date', 'date_time',
                            'event_type', 'penalty_severity', 'penalty_minutes', 'event_team',
                            'event_team_type', 'period_type', 'period', 'period_seconds',
                            'period_seconds_remaining',
                            'strength_state', 'strength', 'empty_net',
                            'extra_attacker', 'home_skaters', 'away_skaters', 'order',
                            ]

    df_train.drop(columns=drop_columns, inplace=True)
    df_test.drop(columns=drop_columns, inplace=True)

    # Create win column
    df_train['win'], df_test['win'] = (df_train['home_final'] > df_train['away_final']).astype(int), (df_test['home_final'] > df_test['away_final']).astype(int)
    
    # Get all the names of the teams in the 'team_catboost_files' directory
    files = os.listdir('team_catboost_files')
    
    # Iterate through each team
    pbar = tqdm(total=len(df_last_two['home_name'].unique()))
    for team in df_last_two['home_name'].unique():
        if team == 'American All-Stars':
            continue
        
        # Check if the team already has a model
        if team in files:
            pbar.update(1)
            pbar.set_description(f'{team} - Already exists')
            continue
        
        else:
            res = get_catboost_and_pickle(team, df_train, df_test)
            pbar.update(1)
            pbar.set_description(f'{team} - {res}')