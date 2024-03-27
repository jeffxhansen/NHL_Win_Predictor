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

if __name__ == "__main__":
    args = sys.argv[1:]  # Read command-line arguments, excluding script name
    if len(args) != 2:
        print("Error: Please provide n_games and save_path arguments.")
        print("Usage: python script.py n_games save_path")
    else:
        try:
            n_games = int(args[0])
            save_path = args[1]
            simulate_games(n_games, save_path)
        except ValueError:
            print("Error: n_games must be an integer.")
        
    
            
            
        

    
    
    

