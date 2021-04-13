import io
from pathlib import Path
import pandas as pd
import numpy as np
from keras import regularizers
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.model_selection import train_test_split as splitter
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
    Path("exploratory").mkdir(parents=True, exist_ok=True)

    with open('exploratory/' + dataset + '_output.txt', "w",
              encoding="utf-8") as f:
        f.write("Label \n")
        f.write(str(data['Label'].value_counts().values))
        f.write("\nRows \n")
        f.write(str(data.shape[0]))
        f.write(" \nCoulmns \n")
        f.write(str(len(data.columns)))
        f.write("\n")
        f.write(str(data.columns))

    for column in data.columns:
        if data[column].dtype == type(object):
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])
    y = data.Label
    x = data.drop('Label', axis=1)
    label_number = len(set(y))
    print(label_number)
    profile = cProfile.Profile()
    x_train, x_test, y_train, y_test = splitter(x, y, test_size=0.3)
    profile.enable()
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    val_indices = 200
    x_val = x_train[-val_indices:]
    y_val = y_train[-val_indices:]
    # train and test
    model = Sequential()
    model.add(Dense(100, activation='relu', input_dim=x_train.shape[1], kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.3))
    model.add(Dense(25, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(25, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.3))
    model.add(Dense(10, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(10, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(label_number, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=50, batch_size=512, validation_data=(x_val, y_val))
    y_pred = model.predict(x_test)
    profile.disable()
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)
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
    # model.save('savemodel/' + dataset + '.h5')


if __name__ == "__main__":
    # Set the correct data set path
    data = pd.read_csv('../dos_fin.csv', delimiter=',')
    train_and_test('keras_dos_fin', data)
    data = pd.read_csv('../dos_rpau.csv', delimiter=',')
    train_and_test('keras_dos_rpau', data)
    data = pd.read_csv('../dos_scanning_attacks.csv', delimiter=',')
    train_and_test('keras_dos_scanning_attacks', data)
    data = pd.read_csv('../dos_sync.csv', delimiter=',')
    train_and_test('keras_dos_sync', data)
    data = pd.read_csv('../scanning_nmap1.csv', delimiter=',')
    train_and_test('keras_scanning_nmap1', data)
    data = pd.read_csv('../scanning_nmap2.csv', delimiter=',')
    train_and_test('keras_scanning_nmap2', data)
    data = pd.read_csv('../scanning_nmap3.csv', delimiter=',')
    train_and_test('keras_scanning_nmap3', data)
    data = pd.read_csv('../dos_rpau.csv', delimiter=',')
    train_and_test('keras_dos_rpau.csv', data)
    data = pd.read_csv('../dos_rpau.csv', delimiter=',')
    train_and_test('keras_dos_rpau.csv', data)
    data = pd.read_csv('../dos_rpau.csv', delimiter=',')
    train_and_test('keras_dos_rpau.csv', data)



