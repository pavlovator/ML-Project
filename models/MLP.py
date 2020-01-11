import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam
from keras import backend as K
from keras.utils import to_categorical


def MLP_bow():
    train = pd.read_csv('data/bow_train.csv')
    test = pd.read_csv('data/bow_test.csv')
    val = pd.read_csv('data/bow_val.csv')
    X_train, Y_train = train[[str(i) for i in range(1000)]].values, to_categorical(train['genre'].values)
    X_val, Y_val = val[[str(i) for i in range(1000)]].values, to_categorical(val['genre'].values)
    X_test, Y_test = test[[str(i) for i in range(1000)]].values, to_categorical(test['genre'].values)

    mlp = Sequential()
    mlp.add(Dense(50, activation='relu', input_dim=X_train.shape[1]))
    mlp.add(Dense(100, activation='relu'))
    mlp.add(Dense(100, activation='relu'))
    mlp.add(Dense(100, activation='relu'))
    mlp.add(Dense(Y_train.shape[1], activation='softmax'))

    mlp.compile(loss='mse', optimizer=Adam(lr=0.00005),  metrics=['accuracy'])

    history = mlp.fit(X_train, Y_train, epochs=20, batch_size=350, validation_data=(X_val, Y_val), verbose=2)
    _, test_acc = mlp.evaluate(X_test,Y_test, verbose = 0)
    print("Test accucacy:{:}".format(test_acc))
    mlp.save("model_files/mlp_bow.h5")

    plt.figure()
    plt.plot(history.history['acc'], label='train accuracy')
    plt.plot(history.history['val_acc'], label='validation accuracy')
    plt.legend(loc='best')
    plt.show()

def MLP_bow_pca100():
    train = pd.read_csv('data/bow_pca100_train.csv')
    test = pd.read_csv('data/bow_pca100_test.csv')
    val = pd.read_csv('data/bow_pca100_val.csv')
    X_train, Y_train = train[[str(i) for i in range(100)]].values, to_categorical(train['genre'].values)
    X_val, Y_val = val[[str(i) for i in range(100)]].values, to_categorical(val['genre'].values)
    X_test, Y_test = test[[str(i) for i in range(100)]].values, to_categorical(test['genre'].values)

    mlp = Sequential()
    mlp.add(Dense(50, activation='relu', input_dim=X_train.shape[1]))
    mlp.add(Dense(75, activation='relu'))
    mlp.add(Dense(50, activation='relu'))
    mlp.add(Dense(Y_train.shape[1], activation='softmax'))

    mlp.compile(loss='mse', optimizer=Adam(lr=0.005), metrics=['accuracy'])

    history = mlp.fit(X_train, Y_train, epochs=10, batch_size=500, validation_data=(X_val, Y_val), verbose=2)
    _, test_acc = mlp.evaluate(X_test, Y_test, verbose=0)
    print("Test accucacy:{:}".format(test_acc))
    mlp.save("model_files/mlp_bow_pac100.h5")

    plt.figure()
    plt.plot(history.history['acc'], label='train accuracy')
    plt.plot(history.history['val_acc'], label='validation accuracy')
    plt.legend(loc='best')
    plt.show()

def MLP_bow_pca10():
    train = pd.read_csv('data/bow_pca10_train.csv')
    test = pd.read_csv('data/bow_pca10_test.csv')
    val = pd.read_csv('data/bow_pca10_val.csv')
    X_train, Y_train = train[[str(i) for i in range(10)]].values, to_categorical(train['genre'].values)
    X_val, Y_val = val[[str(i) for i in range(10)]].values, to_categorical(val['genre'].values)
    X_test, Y_test = test[[str(i) for i in range(10)]].values, to_categorical(test['genre'].values)


    mlp = Sequential()
    mlp.add(Dense(50, activation='relu', input_dim=X_train.shape[1]))
    mlp.add(Dense(80, activation='relu'))
    mlp.add(Dense(70, activation='relu'))
    mlp.add(Dense(Y_train.shape[1], activation='softmax'))

    mlp.compile(loss='mse', optimizer=Adam(lr=0.0005),  metrics=['accuracy'])

    history = mlp.fit(X_train, Y_train, epochs=20, batch_size=300, validation_data=(X_val, Y_val), verbose=2)
    _, test_acc = mlp.evaluate(X_test,Y_test, verbose = 0)
    print("Test accucacy:{:}".format(test_acc))
    mlp.save("model_files/mlp_bow_pac10.h5")

    plt.figure()
    plt.plot(history.history['acc'], label='train accuracy')
    plt.plot(history.history['val_acc'], label='validation accuracy')
    plt.legend(loc='best')
    plt.show()

def MLP_tf_idf():
    train = pd.read_csv('data/tf_idf_train.csv')
    test = pd.read_csv('data/tf_idf_test.csv')
    val = pd.read_csv('data/tf_idf_val.csv')
    X_train, Y_train = train[[str(i) for i in range(1000)]].values, to_categorical(train['genre'].values)
    X_val, Y_val = val[[str(i) for i in range(1000)]].values, to_categorical(val['genre'].values)
    X_test, Y_test = test[[str(i) for i in range(1000)]].values, to_categorical(test['genre'].values)

    mlp = Sequential()
    mlp.add(Dense(100, activation='relu', input_dim=X_train.shape[1]))
    mlp.add(Dense(200, activation='relu'))
    mlp.add(Dense(200, activation='relu'))
    mlp.add(Dense(Y_train.shape[1], activation='softmax'))

    mlp.compile(loss='mse', optimizer=Adam(lr=0.00005), metrics=['accuracy'])

    history = mlp.fit(X_train, Y_train, epochs=20, batch_size=350, validation_data=(X_val, Y_val), verbose=2)
    _, test_acc = mlp.evaluate(X_test, Y_test, verbose=0)
    print("Test accucacy:{:}".format(test_acc))
    mlp.save("model_files/mlp_tf_idf.h5")

    plt.figure()
    plt.plot(history.history['acc'], label='train accuracy')
    plt.plot(history.history['val_acc'], label='validation accuracy')
    plt.legend(loc='best')
    plt.show()

def MLP_tf_idf_pca100():
    train = pd.read_csv('data/tf_idf_pca100_train.csv')
    test = pd.read_csv('data/tf_idf_pca100_test.csv')
    val = pd.read_csv('data/tf_idf_pca100_val.csv')
    X_train, Y_train = train[[str(i) for i in range(100)]].values, to_categorical(train['genre'].values)
    X_val, Y_val = val[[str(i) for i in range(100)]].values, to_categorical(val['genre'].values)
    X_test, Y_test = test[[str(i) for i in range(100)]].values, to_categorical(test['genre'].values)

    mlp = Sequential()
    mlp.add(Dense(75, activation='relu', input_dim=X_train.shape[1]))
    mlp.add(Dense(125, activation='relu'))
    mlp.add(Dense(125, activation='relu'))
    mlp.add(Dense(Y_train.shape[1], activation='softmax'))

    mlp.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])

    history = mlp.fit(X_train, Y_train, epochs=15, batch_size=300, validation_data=(X_val, Y_val), verbose=2)
    _, test_acc = mlp.evaluate(X_test, Y_test, verbose=0)
    print("Test accucacy:{:}".format(test_acc))
    mlp.save("model_files/mlp_tf_idf_pca100.h5")

    plt.figure()
    plt.plot(history.history['acc'], label='train accuracy')
    plt.plot(history.history['val_acc'], label='validation accuracy')
    plt.legend(loc='best')
    plt.show()

def MLP_tf_idf_pca10():
    train = pd.read_csv('data/tf_idf_pca10_train.csv')
    test = pd.read_csv('data/tf_idf_pca10_test.csv')
    val = pd.read_csv('data/tf_idf_pca10_val.csv')
    X_train, Y_train = train[[str(i) for i in range(10)]].values, to_categorical(train['genre'].values)
    X_val, Y_val = val[[str(i) for i in range(10)]].values, to_categorical(val['genre'].values)
    X_test, Y_test = test[[str(i) for i in range(10)]].values, to_categorical(test['genre'].values)

    mlp = Sequential()
    mlp.add(Dense(50, activation='relu', input_dim=X_train.shape[1]))
    mlp.add(Dense(80, activation='relu'))
    mlp.add(Dense(70, activation='relu'))
    mlp.add(Dense(Y_train.shape[1], activation='softmax'))

    mlp.compile(loss='mse', optimizer=Adam(lr=0.0005), metrics=['accuracy'])

    history = mlp.fit(X_train, Y_train, epochs=20, batch_size=300, validation_data=(X_val, Y_val), verbose=2)
    _, test_acc = mlp.evaluate(X_test, Y_test, verbose=0)
    print("Test accucacy:{:}".format(test_acc))
    mlp.save("model_files/mlp_tf_idf_pca10.h5")

    plt.figure()
    plt.plot(history.history['acc'], label='train accuracy')
    plt.plot(history.history['val_acc'], label='validation accuracy')
    plt.legend(loc='best')
    plt.show()

#MLP_bow() #0.5830257148406446
#MLP_bow_pca100() #0.5651004191879366
#MLP_bow_pca10() #0.530125758644637
#MLP_tf_idf() #0.5853719577274371
#MLP_tf_idf_pca100() #0.5605643496140144
#MLP_tf_idf_pca10() #0.5458299443158355