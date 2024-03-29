import pandas as pd
import pickle
from tqdm import tqdm
import numpy as np
import os
import sys

def simulate_games(n_games:int, save_path:str):
    
    # if the save_path is not a valid save path, raise a ValueError
    dir_name = os.path.dirname(save_path)
    if not os.path.exists(dir_name):
        raise ValueError(
            f"{dir_name} does not exist, so save_path={save_path} not valid")

    # load the average probability table
    probs = pd.read_csv("data/probabilities_avg_NOHIT.csv")

    # more or less normalize the probabilities
    probs_sums = (probs
        .copy()
        .groupby(by=['prev3', 'prev2', 'prev1'])
        [['probability_avg', 'probability_1', 'probability_2', 'probability_3']]
        .sum()
        .reset_index()
        .rename(columns={
            'probability_avg': 'probability_avg_sum',
            'probability_1': 'probability_1_sum',
            'probability_2': 'probability_2_sum',
            'probability_3': 'probability_3_sum'}))

    probs2 = pd.merge(probs, probs_sums, on=['prev3', 'prev2', 'prev1'], how='left')
    probs2['probability_avg'] /= probs2['probability_avg_sum']
    probs2['probability_1'] /= probs2['probability_1_sum']
    probs2['probability_2'] /= probs2['probability_2_sum']
    probs2['probability_3'] /= probs2['probability_3_sum']
    probs2 = probs2.drop(columns=[
        'probability_avg_sum', 'probability_1_sum', 'probability_2_sum', 'probability_3_sum'])

    probs2 = probs2.fillna(0)
    
    # get a list of the possible events
    events = probs2['curr_event'].unique()

    # sample 100000 times from the saved NOHIT kde
    kde_nohit_path = "data/pickles/kde_seconds_NOHIT.pickle"
    with open(kde_nohit_path, 'rb') as file:
        kde = pickle.load(file)
    
    ####################################
    #        SIMULATE THE GAME         #
    ####################################
    
    games = []

    # simulte n_games
    game_bar = tqdm(total=n_games)
    for game_id in range(n_games):
        
        i = 0
        samples = kde.sample(100000).astype(int).flatten()
        
        # setup the start of the game
        game_id = str(game_id).zfill(8)
        seconds_remaining = 3600
        prev3 = "#"
        prev2 = "#"
        prev1 = "#"
        home_score = 0
        away_score = 0
        
        # the curr_dict will keep track of the current state of the game
        # with counts of each event
        curr_dict = {e:0 for e in events}
        curr_dict['time_remaining'] = seconds_remaining
        game_dicts = [curr_dict.copy()]
        
        while seconds_remaining > 0:
            
            game_bar.set_description(str(seconds_remaining))
            
            # select a next event based on the probabilities
            curr_table = probs2[(probs['prev3'] == prev3) & (probs2['prev2'] == prev2) & (probs2['prev1'] == prev1)]
            curr_event = np.random.choice(curr_table['curr_event'], p=curr_table['probability_avg'])
            prev3, prev2, prev1 = prev2, prev1, curr_event
            
            # sample from the distribution of how long it takes for an
            # event to occur, and upated the seconds_remaining
            event_time = samples[i]
            i += 1
            seconds_remaining -= event_time
            
            # update the state dictionary
            curr_dict['time_remaining'] = seconds_remaining
            curr_dict[curr_event] += 1
            game_dicts.append(curr_dict.copy())
                
        game_bar.update(1)
            
        # create this games dataframe and add it to the list of dataframes
        game_df = pd.DataFrame(game_dicts)
        game_df['game_id'] = game_id
        game_df['home_score'] = home_score
        game_df['away_score'] = away_score
        games.append(game_df)
        
        # temporarily saves games in case something crashes
        with open("data/pickles/temp_games3.pickle", 'wb') as file:
            pickle.dump(games, file)
    
    # put all of the simulated games into one dataframe
    games_full = pd.concat(games).reset_index(drop=True)
    
    # save the dataframe to the save_path
    if '.csv' in save_path:
        games_full.to_csv(save_path)
    elif '.feather' in save_path:
        games_full.to_feather(save_path)
    else:
        raise ValueError("save path must be of type .csv or .feather")

    return games_full

def get_simulation_prob(curr_snapshot:pd.DataFrame, probs:pd.DataFrame, kde, verbose=False):
    
    # extract the events_dictionary, seconds_remaining, and the latest
    # three events from the current game snapshot
    curr_dict = curr_snapshot.to_dict(orient='records')[-1]
    seconds_remaining = curr_dict['game_seconds_remaining']
    
    diffs = (curr_snapshot.diff()
        .dropna()
        .reset_index(drop=True)
        .drop(columns=['game_seconds_remaining']))
    cols = np.array(diffs.columns)
    inds = np.argmax(diffs.to_numpy(), axis=1)
    events = cols[inds]
    if len(events) >= 3:
        prev3, prev2, prev1 = events[-3], events[-2], events[-1]
    elif len(events) == 2:
        prev3, prev2, prev1 = '#', events[-2], events[-1]
    elif len(events) == 1:
        prev3, prev2, prev1 = '#', '#', events[-1]
    else:
        prev3, prev2, prev1 = '#', '#', '#'
        
    start_dict = curr_dict.copy()
    start_seconds_remaining = seconds_remaining
    start_prev3 = prev3
    start_prev2 = prev2
    start_prev1 = prev1
    
    ####################################
    #        SIMULATE THE GAME         #
    ####################################
    
    n_games = 50

    # simulte n_games
    games = []
    last_game_states = []
    game_bar = tqdm(total=n_games) if verbose else None
    for game_id in range(n_games):
        
        curr_dict = start_dict.copy()
        seconds_remaining = start_seconds_remaining
        prev3 = start_prev3
        prev2 = start_prev2
        prev1 = start_prev1
        
        i = 0
        samples = kde.sample(100000).astype(int).flatten()
        
        # setup the start of the game
        game_id = str(game_id).zfill(8)
        
        # the curr_dict will keep track of the current state of the game
        # with counts of each event
        curr_dict['time_remaining'] = seconds_remaining
        game_dicts = [curr_dict.copy()]
        
        while ((seconds_remaining > 0) or 
               (curr_dict['GOAL_HOME'] == curr_dict['GOAL_AWAY'])):
            
            game_bar.set_description(str(seconds_remaining)) if verbose else None
            
            # select a next event based on the probabilities
            curr_table = probs[(probs['prev3'] == prev3) & (probs['prev2'] == prev2) & (probs['prev1'] == prev1)]
            curr_event = np.random.choice(curr_table['curr_event'], p=curr_table['probability'])
            prev3, prev2, prev1 = prev2, prev1, curr_event
            
            # sample from the distribution of how long it takes for an
            # event to occur, and upated the seconds_remaining
            event_time = samples[i]
            i += 1
            seconds_remaining -= event_time
            
            # update the state dictionary
            curr_dict['time_remaining'] = seconds_remaining
            curr_dict[curr_event] += 1
            game_dicts.append(curr_dict.copy())
                
        game_bar.update(1) if verbose else None
            
        # create this games dataframe and add it to the list of dataframes
        last_game_states.append(curr_dict.copy())
    
    final_states_df = pd.DataFrame(last_game_states)
    home_wins = np.sum(final_states_df['GOAL_HOME'] > final_states_df['GOAL_AWAY'])
    home_prob = home_wins / n_games
    away_prob = 1 - home_prob

    return home_prob, away_prob
        
    
            
            
        

    
    
    

