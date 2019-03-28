import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def main(mode):
    df = pd.read_csv('data/input_train.csv')
    data_other = df.drop(columns=['peak_load','index','Taipei','newTaipei','taichung',\
    'kaohsiung' ,'taoyuan',	'tainan', 'changhua', 'pingtung', 'yunlin', 'hsinchu'],axis=1)
    data_peak  = df['peak_load']

    # split data into 70% training and 30% testing
    train_other, test_other, train_peak, test_peak = train_test_split(data_other, data_peak, test_size=0.3)
    
    # standardize data by feature scaling
    sc = StandardScaler()
    train_other = sc.fit_transform(train_other)
    test_other = sc.fit_transform(test_other)
    
    # add classifier and train model
    support_vector_classifier = SVC(kernel='rbf')
    support_vector_classifier.fit(train_other,train_peak)

    if (mode == "train"):
        # predict
        predict_peak = support_vector_classifier.predict(test_other)

        # calculate error
        # percentage deviation : arithmetic mean
        # average deviation of testing set : RMSE 
        test_peakList = test_peak.tolist()
        sumRate = 0
        sumRMSE = 0
        for i in range(len(test_peakList)):
            dist = abs(predict_peak[i]-test_peakList[i])
            errRate = dist/test_peakList[i]
            sumRate += errRate
            sumRMSE += dist*dist
        
        print('\n\npercentage\t', sumRate/len(test_peakList))
        print('RMSE\t\t', (sumRMSE/len(test_peakList))**(1/2))
    
    else:
        # load forecasting data as predict material
        input_predict = pd.read_csv('data/input_predict.csv')
        input_predict = input_predict.drop(columns=['peak_load','index','Taipei','newTaipei','taichung',\
        'kaohsiung' ,'taoyuan',	'tainan', 'changhua', 'pingtung', 'yunlin', 'hsinchu'],axis=1)
        
        # standardize data by feature scaling
        input_predict = sc.fit_transform(input_predict)

        predict_peak = support_vector_classifier.predict(input_predict)
        print('\n\n', predict_peak)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='predict', 
    help='choose training mode (measure accuracy) or predicting mode (by input_predict.csv), default predict')
    args = parser.parse_args()
    main(args.mode)