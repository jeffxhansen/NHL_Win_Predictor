import pandas as pd
import pickle
from tqdm import tqdm
import numpy as np
import os
import sys
import time
from jeffutils.utils import stack_trace

from py_files.simulator import get_simulation_prob

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

def get_game_probabilities(game_id):
    
    df_full = pd.read_feather("data/play_by_play/play_by_play_full_state_space.feather")
    
    # load the probability table
    prob_mc = pd.read_csv("data/probability_tables/probabilities_avg_NOHIT.csv")
    prob_mc = prob_mc[['prev3', 'prev2', 'prev1', 'curr_event', 'probability_avg']]
    prob_mc = prob_mc.rename(columns={'probability_avg': 'probability'})

    df = df_full.copy()

    curr_game = df.loc[df['game_id'] == game_id, :].copy()
    home_name = curr_game['home_name'].values[0]
    away_name = curr_game['away_name'].values[0]
    game_date = curr_game['game_date'].values[0]

    simulation_cols = ['game_seconds_remaining'] + [c for c in curr_game.columns if 'STATE' in c]
    simulation_cols_new = [c.replace("STATE_", "") for c in simulation_cols]
    state_portion = (curr_game
        .loc[:, simulation_cols]
        .copy()
        .reset_index(drop=True)
        .rename(columns=dict(zip(simulation_cols, simulation_cols_new))))

    # load the probability table
    prob_mc = pd.read_csv("data/probability_tables/probabilities_avg_NOHIT.csv")
    prob_mc = prob_mc[['prev3', 'prev2', 'prev1', 'curr_event', 'probability_avg']]
    prob_mc = prob_mc.rename(columns={'probability_avg': 'probability'})

    with open("data/pickles/kde_seconds_NOHIT.pickle", "rb") as f:
        kde_seconds = pickle.load(f)
        
    # sample from the game every minute
    rows_to_simulate = list(range(0, len(state_portion), len(state_portion) // 10))
    simulation_probabilities = []
    for i, ind in enumerate(rows_to_simulate):
        inds = np.arange(max(0, ind-4), ind+1)
        curr_snapshot = state_portion.loc[inds, :].copy()
        print("Seconds remaining:", curr_snapshot['game_seconds_remaining'].values[-1], f"({i+1}/{len(rows_to_simulate)})")
        
        home_prob, away_prob = get_simulation_prob(curr_snapshot, prob_mc, kde_seconds, verbose=True)
        simulation_probabilities.append((home_prob, away_prob))
        
    save_stuff = {
        'game_id': game_id,
        'curr_game': curr_game,
        'rows_to_simulate': rows_to_simulate,
        'simulation_probabilities': simulation_probabilities
    }
    
    with open(f"data/pickles/simulation_probs_{game_id}.pickle", "wb") as f:
        pickle.dump(save_stuff, f)
        
def simulate_games():
    game_ids = [2022020839, 2023020412, 2022020727, 2022030154, 
                2022021296, 2023020164, 2022020793, 2023020317, 
                2022020971, 2022030145, 2022020938, 2022021300, 
                2022021101, 2022020747, 2023020254, 2022020832, 
                2022020842, 2022021012, 2023020268, 2022020867, 
                2022030324, 2023020546, 2022021001, 2022021055, 
                2023020464, 2022021181, 2023020096, 2022020688, 
                2023020325, 2022021166, 2022020932, 2022020843, 
                2023020365, 2022030245, 2022021040, 2022021088, 
                2023020570, 2023020264, 2023020129, 2022020876, 
                2023020292, 2023020010, 2022021270, 2022021115, 
                2022020753, 2023020475, 2023020192, 2022020662, 
                2023020033, 2023020482]
    
    already_done_game_ids = set()
    
    directory = "data/pickles"
    for file_name in os.listdir(directory):
        path = os.path.join(directory, file_name)
        if os.path.exists(path) and "simulation_probs" in file_name:
            game_id = int(file_name.split("_")[-1].split(".")[0])
            already_done_game_ids.add(game_id)
            
    game_ids = [i for i in game_ids if i not in already_done_game_ids]
    
    mean_time = 10*60
    times = []
    for i, game_id in enumerate(game_ids):
        n_left = len(game_ids) - i
        time_left = (n_left * mean_time) / 60
        print(game_id, f"({i+1}/{len(game_ids)}) ~{round(time_left, 2)} min left")
        how_long = time.time()
        get_game_probabilities(game_id)
        how_long = time.time() - how_long
        times.append(how_long)
        
        mean_time = np.mean(times)
    
if __name__ == "__main__":
    simulate_games()
    '''args = sys.argv[1:]  # Read command-line arguments, excluding script name
    if len(args) != 1:
        print("Error: Please provide game_id.")
        print("Usage: python script.py game_Id")
    else:
        try:
            game_id = int(args[0])
            get_game_probabilities(game_id)
        except Exception as e:
            print(stack_trace(e))'''
        
    
            
            
        

    
    
    

