import pandas as pd

def main():
    current_gameweek = 30

    #fetch csv file from system
    df = pd.read_csv("../../resources/player_points.csv")

    #algorithm to split into sets of 5 gameweek columns and 1 points column
    df_id = df.iloc[:,:3]

    df_training_data = pd.DataFrame()

    i=0

    for i in range(1,current_gameweek-5):
        df_5_weeks = df.iloc[:,4*(i-1)+3:4*(i-1)+23]
        df_week_6_points = df.iloc[:,4*(i-1)+26:4*(i-1)+27]

        df_week_6_classification = df_week_6_points.copy()

        # REGRESSION
        # df_total = pd.concat([df_5_weeks,df_week_6_points],axis=1)

        # Classification
        gameweek_string = "gameweek_" + str(i+5) + "_points"
        # Order of functions is very important here
        df_week_6_classification.loc[df_week_6_classification["gameweek_" + str(i+5) + "_points"] < 4,gameweek_string] = 0
        df_week_6_classification.loc[df_week_6_classification["gameweek_" + str(i+5) + "_points"] >= 8, gameweek_string] = 2
        df_week_6_classification.loc[df_week_6_classification["gameweek_" + str(i+5) + "_points"] >= 4, gameweek_string] = 1

        df_total = pd.concat([df_5_weeks,df_week_6_classification],axis=1)

        # appending to total training data
        end_list = df_total.values.tolist()
        df_training_data = df_training_data.append([pd.DataFrame(end_list)],ignore_index=True)

        # writing 5 gameweeks of prediction data to file
        df_prediction = pd.concat([df_id,df_5_weeks],axis=1)
        df_prediction.to_csv("../../resources/prediction_data/gameweek_" + str(i+5) + "_prediction_data.csv",index=False)

        # writing points at each of the gameweeks
        df_points = pd.concat([df_id,df_week_6_classification],axis=1)
        df_points.to_csv("../../resources/points_data/gameweek_" + str(i+5) + "_points_data.csv",index=False)

    df_training_data.to_csv("../../resources/training_data.csv", index=False, header=False)

    # For current gameweek
    i += 1

    df_5_weeks = df.iloc[:, 4 * (i - 1) + 3:4 * (i - 1) + 23]
    df_prediction = pd.concat([df_id, df_5_weeks], axis=1)
    df_prediction.to_csv("../../resources/prediction_data/gameweek_" + str(i + 5) + "_prediction_data.csv", index=False)

    #algorithm to extract gameweek data for ML model from previous set



if __name__=="__main__":
    main()