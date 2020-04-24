import pandas as pd
from imblearn.combine import SMOTETomek
from collections import Counter
from sklearn.linear_model import LogisticRegression

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

def main():
    df = pd.read_csv("../../resources/training_data.csv",header=None)
    df_x = df.iloc[:,:-1]
    df_y = df.iloc[:,-1]

    print("Before resampling: " + str(Counter(df_y)))

    x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.33,random_state=1)

    rf = RandomForestClassifier(n_estimators=100)
    nb = GaussianNB()
    lr = LogisticRegression(random_state=0)

    rf.fit(x_train,y_train)
    nb.fit(x_train,y_train)
    lr.fit(x_train,y_train)

    pred_rf = rf.predict(x_test)
    pred_nb = nb.predict(x_test)
    pred_lr = lr.predict(x_test)

    print("Accuracy of RF: " + str(accuracy_score(y_test,pred_rf)))
    print("Accuracy of NB: " + str(accuracy_score(y_test, pred_nb)))
    print("Accuracy of LR: " + str(accuracy_score(y_test, pred_lr)))

    print()

    # USING IMBALANCE PACKAGE

    smt = SMOTETomek(random_state=1)
    df_x_res,df_y_res = smt.fit_resample(df_x,df_y)

    print("After resampling: " + str(Counter(df_y_res)))

    x_res_train, x_res_test, y_res_train, y_res_test = train_test_split(df_x_res,df_y_res,test_size=0.33,random_state=1)

    rf_res = RandomForestClassifier(n_estimators=100)
    nb_res = GaussianNB()
    lr_res = LogisticRegression(random_state=0)

    rf_res.fit(x_res_train,y_res_train)
    nb_res.fit(x_res_train,y_res_train)
    lr_res.fit(x_res_train, y_res_train)

    pred_rf_res = rf_res.predict(x_res_test)
    pred_nb_res = nb_res.predict(x_res_test)
    pred_lr_res = lr_res.predict(x_res_test)

    y_res_test_values = y_res_test.values

    print("Accuracy of RF_res: " + str(accuracy_score(y_res_test,pred_rf_res)))
    print("Accuracy of NB_res: " + str(accuracy_score(y_res_test, pred_nb_res)))
    print("Accuracy of LR_res: " + str(accuracy_score(y_res_test, pred_lr_res)))

    #
    # results = pd.DataFrame()
    # results['Random Forest']=pred_rf
    # results['Naive Bayes']=pred_nb
    # results['True Results']=y_test.values

if __name__=="__main__":
    main()