import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def get_data(df, game_id):
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
    event_types = np.append(event_types, ['TIME_REMAINING', 'HOME', 'WIN', 'TEAM', 'GAME_ID'])

    # Create dictionaries to store event counts for home and away teams
    home_dict = {event: 0 for event in event_types}
    away_dict = {event: 0 for event in event_types}

    # Create DataFrames to store event data for home and away teams
    home_df = pd.DataFrame(columns=home_dict.keys())
    away_df = pd.DataFrame(columns=away_dict.keys())
    
    home_df_rows = []
    away_df_rows = []

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
            home_df_rows.append(home_dict)
            #home_df = home_df.append(home_dict, ignore_index=True)
        else:
            away_dict[row['event_type']] += 1
            away_dict['TIME_REMAINING'] = row['game_seconds_remaining']
            away_dict['HOME'] = 0
            away_dict['WIN'] = 1 if row['home_final'] < row['away_final'] else 0
            away_dict['TEAM'] = row['team_encoded']
            away_dict['GAME_ID'] = game_id
            away_df_rows.append(away_dict)
            #away_df = away_df.append(away_dict, ignore_index=True)
            
    home_df = pd.DataFrame(home_df_rows, columns=home_dict.keys())
    away_df = pd.DataFrame(away_df_rows, columns=away_dict.keys())
        
    return home_df, away_df


def get_label_encoder(teams):
    le = LabelEncoder()
    le.fit(teams)
    return le


def plot_game(team, team_id):
    """
    Plot the data for a given team.

    Parameters:
        team (DataFrame): The data for the team to be plotted.
        team_id (int): The ID of the team.

    Returns:
        None: This function displays the plot but does not return any value.
    """

    # Get the columns we need to plot.
    plot_names = team.columns
    
    # Get the number of rows and columns necessary for the plot
    num_cols = 3
    num_rows = len(plot_names) // num_cols + 1 if len(plot_names) % num_cols != 0 else len(plot_names) // num_cols
    
    # Create the plot
    fig, ax = plt.subplots(num_rows, num_cols, figsize=(10, 10))
    
    # Get x-axis values
    x = range(len(team))
    
    # Iterate through the columns and plot them
    for i, column in enumerate(plot_names):
        row = i // num_cols
        col = i % num_cols
        ax[row, col].plot(x, team[column])
        ax[row, col].set_title(column)
        ax[row, col].set_xlabel('Time')
        ax[row, col].set_ylabel('Count')
    
    # Set the title
    fig.suptitle(label_encoder.inverse_transform([team_id])[0])  # Assuming label_encoder is defined elsewhere
    plt.tight_layout()
    plt.show()


def plot_hist(team, team_id, normalize=False):
    """
    Plot Histogram Density plots for each column of the team data.

    Parameters:
        team (DataFrame): The data for the team to be plotted.
        team_id (int): The ID of the team.

    Returns:
        None: This function displays the plot but does not return any value.
    """
    
    # Get the columns we need to plot.
    plot_names = team.columns
    
    # Get the number of rows and columns necessary for the plot
    num_cols = 3
    num_rows = len(plot_names) // num_cols + 1 if len(plot_names) % num_cols != 0 else len(plot_names) // num_cols
    
    # Create the plot
    fig, ax = plt.subplots(num_rows, num_cols, figsize=(10, 10))
    
    # Iterate through the columns and plot the KDE for each
    for i, column in enumerate(plot_names):
        row = i // num_cols
        col = i % num_cols
        
        # Plot the histogram
        ax[row, col].hist(team[column], bins=len(team[column].unique())-1, alpha=0.8, color='blue', edgecolor='black', lw=0.96, density=normalize)
        ax[row, col].set_title(column)
        
    # Set the main title of the plot
    if type(team_id) == str:
        fig.suptitle(team_id)
    else:
        plt.suptitle(label_encoder.inverse_transform([team_id])[0])  # Assuming label_encoder is defined elsewhere
    
    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()
