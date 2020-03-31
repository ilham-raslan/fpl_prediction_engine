import pandas as pd

def main():
    current_gameweek = 32

    #fetch csv file from system
    df = pd.read_csv("../../resources/player_points.csv")

    #algorithm to split into sets of 5 gameweek columns and 1 points column
    df_id = df.iloc[:,:3]

    df_training_data = pd.DataFrame()

    for i in range(1,current_gameweek-5):
        df_5_weeks = df.iloc[:,4*(i-1)+3:4*(i-1)+23]
        df_week_6_points = df.iloc[:,4*(i-1)+26:4*(i-1)+27]
        df_total = pd.concat([df_5_weeks,df_week_6_points],axis=1)

        end_list = df_total.values.tolist()
        df_training_data = df_training_data.append([pd.DataFrame(end_list)],ignore_index=True)

    df_training_data.to_csv("../../resources/training_data.csv",index=False,header=False)
    print("Done")
    #algorithm to extract gameweek data for ML model from previous set

if __name__=="__main__":
    main()