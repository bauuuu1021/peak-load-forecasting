import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def main():
    df = pd.read_csv('data/input.csv')
    data_other = df.drop(columns=['peak_load'],axis=1)
    data_other = df.drop(columns=['index'],axis=1)
    data_peak  = df['peak_load']

    # split data into 70% training and 30% testing
    train_other, test_other, train_peak, test_peak = train_test_split(data_other, data_peak, test_size=0.3)
    
    # standardize data by feature scaling
    sc = StandardScaler()
    train_other = sc.fit_transform(train_other)
    test_other = sc.fit_transform(test_other)
    
    # add classifier, train, and predict
    support_vector_classifier = SVC(kernel='rbf')
    support_vector_classifier.fit(train_other,train_peak)
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
    
    print('--------------------')
    print('percentage\t', sumRate/len(test_peakList))
    print('RMSE\t\t', (sumRMSE/len(test_peakList))**(1/2))

if __name__ == "__main__":
    main()