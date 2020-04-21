import pandas as pd
from sklearn.linear_model import LinearRegression

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

def main():
    df = pd.read_csv("../../resources/training_data.csv",header=None)
    df_x = df.iloc[:,:-1]
    df_y = df.iloc[:,-1]

    x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.33,random_state=1)

    # reg = LinearRegression().fit(x_train, y_train)
    # score = reg.score(x_train, y_train)
    #
    # pred_y = reg.predict(x_test)
    # y_test_np = y_test.values
    #
    # print("Done")

    rf = RandomForestClassifier(n_estimators=100)
    nb = GaussianNB()

    rf.fit(x_train,y_train)
    nb.fit(x_train,y_train)

    pred_rf = rf.predict(x_test)
    pred_nb = nb.predict(x_test)

    y_test_values = y_test.values

    print("Accuracy of RF: " + str(accuracy_score(y_test,pred_rf)))
    print("Accuracy of NB: " + str(accuracy_score(y_test, pred_nb)))

    print("Done")

    #
    # results = pd.DataFrame()
    # results['Random Forest']=pred_rf
    # results['Naive Bayes']=pred_nb
    # results['True Results']=y_test.values

if __name__=="__main__":
    main()