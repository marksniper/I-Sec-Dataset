from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split as splitter
from xgboost import XGBClassifier
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
    label_number = len(set(y))
    if(label_number < 2):
        model = XGBClassifier(objective='multi:softprob', booster='gbtree', verbosity=0)
    else:
        model = XGBClassifier( booster='gbtree', verbosity=0)
    model.fit(x_train, y_train, eval_metric='mlogloss')
    y_pred = model.predict(x_test)
    profile.disable()
    profile.dump_stats('output.prof')
    stream = open('result/' + dataset + '_profiling.txt', 'w')
    stats = pstats.Stats('output.prof', stream=stream)
    stats.sort_stats('cumtime')
    stats.print_stats()
    os.remove('output.prof')
    conf_matrix = confusion_matrix(y_test, y_pred)
    f = open('result/' + dataset + '_output.txt', 'w')
    sys.stdout = f
    print(conf_matrix)
    print(classification_report(y_test, y_pred))
    # joblib.dump(model, 'savemodel/'+ dataset +'.model')


if __name__ == "__main__":
    data = pd.read_csv('../dos_fin.csv', delimiter=',')
    train_and_test('xgboost_dos_fin', data)
    data = pd.read_csv('../dos_rpau.csv', delimiter=',')
    train_and_test('xgboost_dos_rpau', data)
    data = pd.read_csv('../dos_scanning_attacks.csv', delimiter=',')
    train_and_test('xgboost_dos_scanning_attacks', data)
    data = pd.read_csv('../dos_sync.csv', delimiter=',')
    train_and_test('xgboost_dos_sync', data)
    data = pd.read_csv('../scanning_nmap1.csv', delimiter=',')
    train_and_test('xgboost_scanning_nmap1', data)
    data = pd.read_csv('../scanning_nmap2.csv', delimiter=',')
    train_and_test('xgboost_scanning_nmap2', data)
    data = pd.read_csv('../scanning_nmap3.csv', delimiter=',')
    train_and_test('xgboost_scanning_nmap3', data)
    # no correlated
    data = pd.read_csv('../no_correlated/dos_fin_training.csv', delimiter=',')
    train_and_test('xgboost_nc_dos_fin_training', data)
    data = pd.read_csv('../no_correlated/dos_rpau_training.csv', delimiter=',')
    train_and_test('xgboost_nc_dos_rpau_training', data)
    data = pd.read_csv('../no_correlated/dos_scanning_merged_training.csv', delimiter=',')
    train_and_test('xgboost_nc_dos_scanning_merged_training.csv', data)
    data = pd.read_csv('../no_correlated/dos_sync_training.csv', delimiter=',')
    train_and_test('xgboost_nc_dos_sync_training', data)
    data = pd.read_csv('../no_correlated/scanning_nmap1_training.csv', delimiter=',')
    train_and_test('xgboost_nc_scanning_nmap1_training', data)
    data = pd.read_csv('../no_correlated/scanning_nmap2_training.csv', delimiter=',')
    train_and_test('xgboost_nc_scanning_nmap2_training.csv', data)
    data = pd.read_csv('../no_correlated/scanning_nmap3_training.csv', delimiter=',')
    train_and_test('xgboost_nc_scanning_nmap3_training', data)