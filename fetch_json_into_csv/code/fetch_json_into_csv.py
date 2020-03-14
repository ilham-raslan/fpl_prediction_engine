import requests
import pandas as pd
import numpy as np

def main():
    current_gameweek = 32

    URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
    player_data = requests.get(URL).json()['elements']

    df = pd.DataFrame(columns=["id","first_name","last_name"])

    df['id'] = [int(i) for i in range(1,len(player_data)+1)]
    df['id'] = df['id'].astype(int)

    for i in range(len(player_data)):
        df['first_name'].iloc[i] = player_data[i]['first_name']
        df['last_name'].iloc[i] = player_data[i]['second_name']

# -----------------------------------DOESNT WORK FOR GAMEWEEK 26 AND ABOVE, GIVES INDEX OUT OF BOUNDS EXCEPTION ------------------
    for i in range(1,current_gameweek):
        gameweek_data = requests.get("https://fantasy.premierleague.com/api/event/" + str(i) + "/live").json()

        goals_conceded_list = np.zeros(len(player_data))
        goals_scored_list = np.zeros(len(player_data))
        assists_list = np.zeros(len(player_data))
        points_list = np.zeros(len(player_data))

#---------------------------------------ERROR IN HERE-----------------------------------------------------------------------------

        for j in range(len(gameweek_data['elements'])):

            try:
                goals_conceded_list[gameweek_data['elements'][j]['id']-1] = gameweek_data['elements'][j]['stats']['goals_conceded']
                goals_scored_list[gameweek_data['elements'][j]['id']-1] = gameweek_data['elements'][j]['stats']['goals_scored']
                assists_list[gameweek_data['elements'][j]['id']-1] = gameweek_data['elements'][j]['stats']['assists']
                points_list[gameweek_data['elements'][j]['id']-1] = gameweek_data['elements'][j]['stats']['total_points']
            except:
                print("Error occured with gameweek " + str(i) + " and element id " + str(gameweek_data['elements'][j]['id']))
#--------------------------------------------------------------------------------------------------------------------------------
        df['gameweek_' + str(i) + '_goals_conceded'] = goals_conceded_list
        df['gameweek_' + str(i) + '_goals_scored'] = goals_scored_list
        df['gameweek_' + str(i) + '_assists'] = assists_list
        df['gameweek_' + str(i) + '_points'] = points_list

    df.to_csv("../csv_files/player_points.csv")
    print("Done")

if __name__=="__main__":
    main()