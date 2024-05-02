#!/bin/bash

# Define the hockey teams as an array
hockey_teams=("Toronto Maple Leafs" "Washington Capitals" "Vancouver Canucks" \
       "San Jose Sharks" "Buffalo Sabres" "New York Rangers" \
       "Pittsburgh Penguins" "Carolina Hurricanes" "Ottawa Senators" \
       "Detroit Red Wings" "St. Louis Blues" "Dallas Stars" \
       "Colorado Avalanche" "Vegas Golden Knights" \
       "Columbus Blue Jackets" "Los Angeles Kings" \
       "Tampa Bay Lightning" "New Jersey Devils" "New York Islanders" \
       "Minnesota Wild" "Arizona Coyotes" "Calgary Flames" \
       "Chicago Blackhawks" "Boston Bruins" "Anaheim Ducks" \
       "Philadelphia Flyers" "Nashville Predators" "Winnipeg Jets" \
       "Florida Panthers" "Montreal Canadiens" "Edmonton Oilers" "Seattle Kraken")

# Iterate over each hockey team name
for team_name in "${hockey_teams[@]}"; do
    # Define your dynamic output file name with the team name
    output_file="catboost_train_${team_name// /_}.txt"

    # Use the dynamic output file name in your sbatch command
    sbatch --job-name="$team_name" --output="$output_file" cat_jobscript "$team_name"

    # Print a message indicating which team's job is submitted
    echo "Submitted job for $team_name"
done

