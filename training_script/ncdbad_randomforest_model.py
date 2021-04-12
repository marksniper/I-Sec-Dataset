from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split as splitter
from sklearn.ensemble import RandomForestClassifier
import cProfile
import pstats
import os
import sys
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

__version__ = "0.1"
__author__ = 'Benedetto Marco Serinelli'


def train_and_test(dataset, data):
    Path("result").mkdir(parents=True, exist_ok=True)
    Path("savemodel").mkdir(parents=True, exist_ok=True)
    f = open('result/' + dataset + '_output.txt', 'w')
    sys.stdout = f
    for column in data.columns:
        if data[column].dtype == type(object):
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])
    y = data.Label
    x = data.drop('Label', axis=1)
    profile = cProfile.Profile()
    x_train, x_test, y_train, y_test = splitter(x, y, test_size=0.3)
    profile.enable()
    # train and test
    regressor = RandomForestClassifier()
    regressor.fit(x_train, y_train)
    y_pred = regressor.predict(x_test)
    profile.disable()
    profile.dump_stats('output.prof')
    stream = open('result/'+dataset+'_profiling.txt', 'w')
    stats = pstats.Stats('output.prof', stream=stream)
    stats.sort_stats('cumtime')
    stats.print_stats()
    os.remove('output.prof')
    conf_matrix = confusion_matrix(y_test, y_pred)
    f = open('result/'+dataset+'_output.txt', 'w')
    sys.stdout = f
    print(conf_matrix)
    print(classification_report(y_test, y_pred))
    # joblib.dump(regressor, 'savemodel/' + dataset + '.model')


if __name__ == "__main__":
    # set the data set path
    data = pd.read_csv('../../dataset evaluation/new_KDD.csv', delimiter=',')
    # pass the correct data set name
    train_and_test('rf_new_KDD', data)
    # data = pd.read_csv('../../dataset evaluation/new_NSL_KDD.csv', delimiter=',')
    # train_and_test('rf_new_NSL_KDD', data)
    # data = pd.read_csv('../../dataset evaluation/Friday-23-02-2018_clean.csv', delimiter=',')
    # train_and_test('rf_Friday-23-02-2018', data)
    # data = pd.read_csv('../../dataset evaluation/Friday-02-03-2018_clean.csv', delimiter=',')
    # train_and_test('rf_Friday-02-03-2018', data)
    # data = pd.read_csv('../../dataset evaluation/Friday-16-02-2018_clean.csv', delimiter=',')
    # train_and_test('rf_Friday-16-02-2018', data)
    # data = pd.read_csv('../../dataset evaluation/Thursday-01-03-2018_clean.csv', delimiter=',')
    # train_and_test('rf_Thursday-01-03-2018', data)
    # data = pd.read_csv('../../dataset evaluation/Thursday-15-02-2018_clean.csv', delimiter=',')
    # train_and_test('rf_Thursday-15-02-2018', data)
    # data = pd.read_csv('../../dataset evaluation/Thursday-22-02-2018_clean.csv', delimiter=',')
    # train_and_test('rf_Thursday-22-02-2018', data)
    # data = pd.read_csv('../../dataset evaluation/Wednesday-14-02-2018_clean.csv', delimiter=',')
    # train_and_test('rf_Wednesday-14-02-2018', data)
    # data = pd.read_csv('../../dataset evaluation/Wednesday-21-02-2018_clean.csv', delimiter=',')
    # train_and_test('rf_Wednesday-21-02-2018', data)
    # data = pd.read_csv('../../dataset evaluation/Wednesday-28-02-2018_clean.csv', delimiter=',')
    # train_and_test('rf_Wednesday-28-02-2018', data)