# NHL_Win_Predictor


# data

This is an empty directory at the moment. I have added a .gitignore to ignore every single file in this directory, so if you absolutely want to make sure that something gets pushed to the GitHub, you will need to:
1. Open .gitignore
2. add `!data/your_file`
3. save the .gitignore
4. Do the normal git add your_file, git commit -m "message", git pull, git push

# ideas

* Probability threshold to get rid of the little guys and then renormalize probabilities
* Take into account penalty/powerplay positions to get more goals
* +3,+2,+1,0,-1,-2,-3 player comparisons to help raise probability of goals
* regression/correlation analysis to see player-diff affect on goal probability
* get rid of more noisy events

# assumptions

* There are rows where there are 0 home_skaters or 0 away_skaters. Most of these occur during period==5, and they appear to happen during shootouts or penalty shots. Right now, I am working on including power play states, and so I am forward filling these rows where there are 0 skaters.
* ignoring the event_type=='STOP': this is a stoppage of play and can be icing, puck in netting, puck in benches, puck in crowd, goalie stopped, etc. and the following event of a penalty or faceoff is usually more descriptive of what happened
* ignoring the event_type=='HIT' since it happens a lot and is not indicative of winning or losing