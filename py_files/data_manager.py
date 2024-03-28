import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from jeffutils.utils import movecol
import os

RELEVANT_COLUMNS = [
    "game_date",
    "date_time",
    "game_id",
    "home_name",
    "away_name",
    "event_type",
    "description",
    "penalty_severity",
    "penalty_minutes",
    "event_team",
    "event_team_type",
    "period_type",
    "period",
    "period_seconds",
    "period_seconds_remaining",
    "game_seconds",
    "game_seconds_remaining",
    "home_score",
    "away_score",
    "home_final",
    "away_final",
    "strength_state",
    "strength_code",
    "strength",
    "empty_net",
    "extra_attacker",
    "home_skaters",
    "away_skaters"
]

EVENTS_IGNORED = [
    'CHANGE', 'DELAYED_PENALTY', 'CHALLENGE', 
    'FAILED_SHOT_ATTEMPT', 'UNKNOWN', 
    'PERIOD_END', 'GAME_SCHEDULED', 'GAME_END',
    'PERIOD_START', 'SHOOTOUT_COMPLETE', 'EARLY_INT_START',
    'EARLY_INT_END', 'EMERGENCY_GOALTENDER',
    'STOP' # this is a stoppage of play and can be icing, puck in netting, puck in benches, puck in crowd, goalie stopped, ect.
]

def load_and_clean_csv(path):
    """ takes in a year as an integer [2010, 2023] and returns a pandas
    dataframe without any nan values."""
    if '.csv' in path:
        df = pd.read_csv(path, encoding='latin1')
    elif '.feather' in path:
        df = pd.read_feather(path)
    else:
        raise ValueError("path must be of type .csv or .feather")
    
    # only keep the relevant columns, and get rid of the events
    # that are not relevant or never really show up
    rel_cols = [col for col in RELEVANT_COLUMNS if col in df.columns]
    df = df[rel_cols]
    df = df[~df['event_type'].isin(EVENTS_IGNORED)]
    
    for col in df.columns:
        # if the column is type string or has mixed data types
        if df[col].dtype == 'object':
            # fill the nans with a "-" string
            df[col].fillna("-", inplace=True)
            # make the whole column have string type. This makes
            # the column have a single data type
            df[col] = df[col].astype(str)
            
        # if the column is numerical, fill the nans with 0
        else:
            df[col].fillna(0, inplace=True)
            
    # include the date_time and game_date columns. Usually game_date has values
    # when date_time is NaN, so fill the NaN date_time entries with the associated
    # value game_time
    if 'game_date' in df.columns and 'date_time' in df.columns:
        df['date_time'] = df['date_time'].fillna(df['game_date'])
    elif 'game_date' in df.columns:
        df['date_time'] = df['game_date']
    elif 'date_time' in df.columns:
        df['game_date'] = df['date_time']
    else:
        df['date_time'] = "-"
        df['game_date'] = "-"
            
    return df

def load_clean_feather(year):
    """ if the feather file exists, load it and return it. Otherwise, return None
    """
    path = f"data/play_by_play_{year}_{year-2000+1}_clean.feather"
    if os.path.exists(path):
        return pd.read_feather(path)
    else:
        return None
    


def get_data(file):
    
    # Get the file
    df = pd.read_csv(file)
    
    # Drop all NaN's in the event_team_type column
    # These are events that are not associated with a team, but more so stoppage of play.
    df = df.dropna(subset=['event_team_type'])
    
    # Get all unique game ids
    unique_game_ids = df.game_id.unique()
    
    # Get the label encoder for the teams
    team_names = np.sort(df.event_team.dropna().unique())
    label_encoder = get_label_encoder(team_names)

    # Label encode the teams
    df['team_encoded'] = label_encoder.transform(df.event_team)

    # Iterate through the games
    final_df = pd.DataFrame()
    game_bar = tqdm(total=len(unique_game_ids))
    for game_id in unique_game_ids:
        final_place = create_state_space(df, game_id)
        final_df = final_df.append(final_place)
        game_bar.update(1)

    return final_df


def create_state_space_opt(path, show_pbar=True):
    """ takes in a path to a csv (ex: 'data/play_by_play/play_by_play_2010_11.csv'
    and returns a pandas dataframe with the cleaned key data where each row is an 
    event with the state space up to that point. Each state space column is like:
    'STATE_FACEOFF_HOME', 'STATE_FACEOFF_AWAY', etc.
    """
    if not os.path.exists(path):
        raise ValueError(f"{path} does not exist")
    
    df = load_and_clean_csv(path)
    
    # ignore all of the HIT events
    df = df.loc[
        (df['event_type'] != 'HIT') & # comment this line out to include HIT events
        (df['event_team_type'] != "-"), :].copy()
    
    # distinguish the events between home and away
    df['event_type'] = df['event_type'].astype(str) + '_' + df['event_team_type'].astype(str).str.upper()

    # sort the dataframe by game_id and game_seconds to ensure vectorized
    # operations are in order
    df = df.sort_values(by=['game_id', 'game_seconds'], ascending=[True, True])

    # add an order column to ensure that the pandas merge works later
    df['order'] = np.arange(len(df))

    # create a smaller dataframe with just the relevant columns for getting the state counts
    count_cols = ['game_id', 'event_type', 'order']
    other_cols = [c for c in df.columns if c not in count_cols]
    df_rel = df[count_cols].copy()
    df_rel['count'] = 1

    # pivot the table on the event_type, so that there is a zero for each event
    # that occured at each timestep and a 0 for all of the others
    piv_tab = (pd.pivot_table(df_rel, 
                            values='count', 
                            index=['game_id', 'order'], 
                            columns=['event_type'])
            .reset_index()
            .fillna(0))

    # create the cumulative counts for all of the new event columns in the pivot table
    event_cols = [c for c in piv_tab.columns if c != 'game_id' and c != 'order']
    pbar = tqdm(total=len(event_cols)) if show_pbar else None
    for col in event_cols:
        pbar.set_description(str(col)) if show_pbar else None
        
        piv_tab[col] = piv_tab[col].astype(int)
        
        # get a cumulative sum
        piv_tab[col+"_cumsum"] = piv_tab[col].cumsum()

        # for the last entries of each game, get the different between cumulative sums
        last_rows = piv_tab['game_id'].shift(-1) != piv_tab['game_id']
        last_rows_df = piv_tab.loc[last_rows, :].copy()
        last_rows_df[col+"_cumsum"] = -1*last_rows_df[col+"_cumsum"]
        last_rows_df[col+"_cumsum_diff"] = last_rows_df[col+"_cumsum"].diff().fillna(0)
        first_cumsum = last_rows_df.loc[last_rows_df.index[0], col+"_cumsum"]
        last_rows_df.loc[last_rows_df.index[0], col+"_cumsum_diff"] = first_cumsum

        # align cumsum_diff with the original dataframe
        piv_tab = pd.merge(piv_tab, last_rows_df[['game_id', 'order', col+"_cumsum_diff"]], on=['game_id', 'order'], how='left')
        piv_tab[col+"_offset"] = piv_tab[col+"_cumsum_diff"].fillna(0)
        piv_tab[col+"_offset2"] = piv_tab[col+"_offset"].shift(1).fillna(0)

        # compute the adjusted column
        piv_tab[col+"_adjusted"] = piv_tab[col].copy() + piv_tab[col+"_offset2"]

        # cumulative sum column'
        piv_tab[col+'_raw'] = piv_tab[col].copy()
        piv_tab[col+'_total'] = piv_tab[col+"_adjusted"].cumsum().astype(int)

        # only keep the new total column (get rid of the intermediate columns)
        piv_tab = movecol(piv_tab, [col+'_total', col+'_raw'], col, 'After')
        piv_tab = piv_tab.drop(columns=[col])
        piv_tab = piv_tab.rename(columns={col+'_total': col})
        drop_cols = [col+'_cumsum', col+'_cumsum_diff', col+'_offset', col+'_offset2', col+"_adjusted"]
        raw_cols = [c for c in piv_tab.columns if 'raw' in c]
        piv_tab = piv_tab.drop(columns=drop_cols+raw_cols)
        
        # rename the event_cols to have state_ at the front
        piv_tab = piv_tab.rename(columns={col: 'STATE_'+col})
        
        pbar.update(1) if show_pbar else None

    df = df.merge(piv_tab, on=['game_id', 'order'], how='left')
    
    return df


def create_state_space(df, game_id):
    """
    Extracts and organizes event data for a specific game.

    Args:
    df (DataFrame): The DataFrame containing the game data.
    game_id (int): The ID of the game to retrieve data for.

    Returns:
    tuple: A tuple containing two DataFrames, one for home team events and one for away team events.
    """
    # Filter the DataFrame to get data for the specified game and drop rows with NaN values in event_team_type
    game1 = df[df.game_id == game_id] 
    game1 = game1.dropna(subset=['event_team_type'])

    # Get unique event types (excluding CHANGE)
    event_types = game1.event_type.unique()
    event_types = np.delete(event_types, np.where(event_types == 'CHANGE'))
    event_types = np.append(event_types, ['TIME_REMAINING', 'HOME', 'WIN', 'TEAM', 'GAME_ID', 'CORSI', 'FENWICK'])

    # Create dictionaries to store event counts for home and away teams
    final_dict = {f'HOME_{event}': 0 for event in event_types}
    away_dict = {f'AWAY_{event}': 0 for event in event_types}
    final_dict.update(away_dict)

    # Create DataFrames to store event data for home and away teams
    final_df = False

    # Iterate through the events in the game and count them
    for _, row in game1.iterrows():
        # Skip events with NaN event_team_type or events of type CHANGE
        if pd.isnull(row['event_team_type']) or row['event_type'] == 'CHANGE':
            continue
        
        # Determine if the event belongs to the home or away team and update counts accordingly
        if row['event_team_type'] == 'home':
            final_dict[f"HOME_{row['event_type']}"] += 1
            final_dict['TIME_REMAINING'] = row['game_seconds_remaining']
            final_dict['HOME_HOME'] = 1
            final_dict['WIN'] = 1 if row['home_final'] > row['away_final'] else 0
            final_dict['HOME_TEAM'] = row['team_encoded']
            final_dict['GAME_ID'] = game_id

            # Get the Corsi and Fenwick for the home team
            if row['strength_code'] == "EV":
                final_dict['HOME_CORSI'] = calculate_corsi(final_dict, 'HOME')
                final_dict['HOME_FENWICK'] = calculate_fenwick(final_dict, 'HOME')
                final_dict['HOME_CORSI_FOR'] = calculate_corsi_for(final_dict['HOME_CORSI'], final_dict['AWAY_CORSI'])
                final_dict['HOME_FENWICK_FOR'] = calculate_fenwick_for(final_dict['HOME_FENWICK'], final_dict['AWAY_FENWICK'])
            
        else:
            final_dict[f"AWAY_{row['event_type']}"] += 1
            final_dict['TIME_REMAINING'] = row['game_seconds_remaining']
            final_dict['AWAY_HOME'] = 0
            final_dict['WIN'] = 0 if row['home_final'] < row['away_final'] else 1
            final_dict['AWAY_TEAM'] = row['team_encoded']
            final_dict['GAME_ID'] = game_id
            
            # Get the Corsi and Fenwick for the away team
            if row['strength_code'] == "EV":
                final_dict['AWAY_CORSI'] = calculate_corsi(final_dict, "AWAY")
                final_dict['AWAY_FENWICK'] = calculate_fenwick(final_dict, "AWAY")
                final_dict['AWAY_CORSI_FOR'] = calculate_corsi_for(final_dict['AWAY_CORSI'], final_dict['HOME_CORSI'])
                final_dict['AWAY_FENWICK_FOR'] = calculate_fenwick_for(final_dict['AWAY_FENWICK'], final_dict['HOME_FENWICK'])
                

        if type(final_df) is bool:
            final_df = pd.DataFrame(final_dict, index=[0])
        else:
            final_df = final_df.append(final_dict, ignore_index=True)
        
    return final_df


def get_label_encoder(teams):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(teams)
    return le


def calculate_corsi(team_dict, h_or_a):
    """
    Calculates the Corsi for a team.

    Args:
    team_dict (dict): A dictionary containing event counts for a team.

    Returns:
    int: The Corsi for the team.
    """
    return team_dict[f'{h_or_a}_SHOT'] + team_dict[f'{h_or_a}_MISSED_SHOT'] + team_dict[f'{h_or_a}_BLOCKED_SHOT']


def calculate_fenwick(team_dict, h_or_a):
    """
    Calculates the Fenwick for a team.

    Args:
    team_dict (dict): A dictionary containing event counts for a team.

    Returns:
    int: The Fenwick for the team.
    """
    return team_dict[f'{h_or_a}_SHOT'] + team_dict[f'{h_or_a}_MISSED_SHOT']


def calculate_corsi_for(team_corsi, opp_corsi):
    """
    Calculates the Corsi For Percentage for a team.

    Args:
    team_corsi (int): The Corsi for the team.
    opp_corsi (int): The Corsi for the opposing team.

    Returns:
    float: The Corsi For Percentage for the team.
    """
    # Check if the denominator is zero to avoid division by zero
    if team_corsi + opp_corsi == 0:
        return 0
    
    return team_corsi / (team_corsi + opp_corsi)


def calculate_fenwick_for(team_fenwick, opp_fenwick):
    """
    Calculates the Fenwick For Percentage for a team.

    Args:
    team_fenwick (int): The Fenwick for the team.
    opp_fenwick (int): The Fenwick for the opposing team.

    Returns:
    float: The Fenwick For Percentage for the team.
    """
    # Check if the denominator is zero to avoid division by zero
    if team_fenwick + opp_fenwick == 0:
        return 0
    
    return team_fenwick / (team_fenwick + opp_fenwick)