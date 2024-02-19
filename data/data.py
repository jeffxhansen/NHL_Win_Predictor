import pandas as pd
import numpy as np
import os

def load_hockey_scraper_pbp():
    """ loads the combined play-by-play data from the hockey_scraper package
    https://www.kaggle.com/datasets/s903124/nhl-playbyplay-data-from-2007
    
    I just download the archive.zip into the data directory and then extracted
    the archive.zip
    """
    
    combined_path = "data/hockey_scraper_data/pbp/combined_pbp.feather"
    
    if os.path.exists(combined_path):
        df = pd.read_feather(combined_path)
        return df 
    
    else:
        
        pbp_path = "data/hockey_scraper_data/pbp"
        
        # combine all of the pbp files into a single dataframe from the 
        # data/hockey_scraper_data/pbp directory
        big_df = []
        # iterate over every filename in the pbp_path directory
        for filename in os.listdir(pbp_path):
            # if the file is a csv file
            if filename.endswith(".csv"):
                print(filename, " "*20, end="\r")
                # read the csv file into a pandas dataframe
                df = pd.read_csv(os.path.join(pbp_path, filename))
                
                # append the dataframe to the big_df
                big_df.append(df)
                
                
        # concatenate all the dataframes in big_df into a single dataframe
        big_df = pd.concat(big_df, ignore_index=True)
        
        big_df.to_feather(combined_path)
        
        return big_df
    
    