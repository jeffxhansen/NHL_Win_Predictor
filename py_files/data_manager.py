import pandas as pd
import numpy as np

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
    for game_id in unique_game_ids:
        home_df, away_df = get_data(df, game_id)
        final_df = final_df.append(home_df)
        final_df = final_df.append(away_df)


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
    home_dict = {event: 0 for event in event_types}
    away_dict = {event: 0 for event in event_types}

    # Create DataFrames to store event data for home and away teams
    home_df = pd.DataFrame(columns=home_dict.keys())
    away_df = pd.DataFrame(columns=away_dict.keys())

    # Iterate through the events in the game and count them
    for _, row in game1.iterrows():
        # Skip events with NaN event_team_type or events of type CHANGE
        if pd.isnull(row['event_team_type']) or row['event_type'] == 'CHANGE':
            continue
        
        # Determine if the event belongs to the home or away team and update counts accordingly
        if row['event_team_type'] == 'home':
            home_dict[row['event_type']] += 1
            home_dict['TIME_REMAINING'] = row['game_seconds_remaining']
            home_dict['HOME'] = 1
            home_dict['WIN'] = 1 if row['home_final'] > row['away_final'] else 0
            home_dict['TEAM'] = row['team_encoded']
            home_dict['GAME_ID'] = game_id
            
            # Get the Corsi and Fenwick for the home team
            if row['strength_code'] == "EV":
                home_dict['CORSI'] = calculate_corsi(home_dict)
                home_dict['FENWICK'] = calculate_fenwick(home_dict)
                home_dict['CORSI_FOR'] = calculate_corsi_for(home_dict['CORSI'], away_dict['CORSI'])
                home_dict['FENWICK_FOR'] = calculate_fenwick_for(home_dict['FENWICK'], away_dict['FENWICK'])
            
            home_df = home_df.append(home_dict, ignore_index=True)
        else:
            away_dict[row['event_type']] += 1
            away_dict['TIME_REMAINING'] = row['game_seconds_remaining']
            away_dict['HOME'] = 0
            away_dict['WIN'] = 1 if row['home_final'] < row['away_final'] else 0
            away_dict['TEAM'] = row['team_encoded']
            away_dict['GAME_ID'] = game_id
            
            # Get the Corsi and Fenwick for the away team
            if row['strength_code'] == "EV":
                away_dict['CORSI'] = calculate_corsi(away_dict)
                away_dict['FENWICK'] = calculate_fenwick(away_dict)
                away_dict['CORSI_FOR'] = calculate_corsi_for(away_dict['CORSI'], home_dict['CORSI'])
                away_dict['FENWICK_FOR'] = calculate_fenwick_for(away_dict['FENWICK'], home_dict['FENWICK'])
                
            away_df = away_df.append(away_dict, ignore_index=True)
        
    return home_df, away_df


def get_label_encoder(teams):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(teams)
    return le


def calculate_corsi(team_dict):
    """
    Calculates the Corsi for a team.

    Args:
    team_dict (dict): A dictionary containing event counts for a team.

    Returns:
    int: The Corsi for the team.
    """
    return team_dict['SHOT'] + team_dict['MISSED_SHOT'] + team_dict['BLOCKED_SHOT']


def calculate_fenwick(team_dict):
    """
    Calculates the Fenwick for a team.

    Args:
    team_dict (dict): A dictionary containing event counts for a team.

    Returns:
    int: The Fenwick for the team.
    """
    return team_dict['SHOT'] + team_dict['MISSED_SHOT']


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