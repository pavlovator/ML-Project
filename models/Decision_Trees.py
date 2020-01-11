from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.externals import joblib

def RF_bow():
    train = pd.read_csv('data/bow_train.csv')
    test = pd.read_csv('data/bow_test.csv')
    val = pd.read_csv('data/bow_val.csv')
    X_train, Y_train = train[[str(i) for i in range(1000)]].values, train['genre'].values
    X_val, Y_val = val[[str(i) for i in range(1000)]].values, val['genre'].values
    X_test, Y_test = test[[str(i) for i in range(1000)]].values, test['genre'].values
    rfc = RandomForestClassifier(n_estimators=33, criterion='entropy')
    rfc.fit(X_train, Y_train)
    print("Traning accuracy for Random Forest: {}".format(rfc.score(X_train, Y_train)))
    print("Validation accuracy for Random Forest: {}".format(rfc.score(X_val, Y_val)))
    print("Testing accuracy for Random Forest: {}".format(rfc.score(X_test, Y_test)))
    joblib.dump(rfc, 'model_files/RF_bow.sav')
    return rfc

def RF_bow_pca100():
    train = pd.read_csv('data/bow_pca100_train.csv')
    test = pd.read_csv('data/bow_pca100_test.csv')
    val = pd.read_csv('data/bow_pca100_val.csv')
    X_train, Y_train = train[[str(i) for i in range(100)]].values, train['genre'].values
    X_val, Y_val = val[[str(i) for i in range(100)]].values, val['genre'].values
    X_test, Y_test = test[[str(i) for i in range(100)]].values, test['genre'].values
    rfc = RandomForestClassifier(n_estimators=33, criterion='entropy')
    rfc.fit(X_train, Y_train)
    print("Traning accuracy for Random Forest: {}".format(rfc.score(X_train, Y_train)))
    print("Validation accuracy for Random Forest: {}".format(rfc.score(X_val, Y_val)))
    print("Testing accuracy for Random Forest: {}".format(rfc.score(X_test, Y_test)))
    joblib.dump(rfc, 'model_files/RF_bow_pac_100.sav')

def RF_bow_pca10():
    train = pd.read_csv('data/bow_pca10_train.csv')
    test = pd.read_csv('data/bow_pca10_test.csv')
    val = pd.read_csv('data/bow_pca10_val.csv')
    X_train, Y_train = train[[str(i) for i in range(10)]].values, train['genre'].values
    X_val, Y_val = val[[str(i) for i in range(10)]].values, val['genre'].values
    X_test, Y_test = test[[str(i) for i in range(10)]].values, test['genre'].values
    rfc = RandomForestClassifier(n_estimators=33, criterion='entropy')
    rfc.fit(X_train, Y_train)
    print("Traning accuracy for Random Forest: {}".format(rfc.score(X_train, Y_train)))
    print("Validation accuracy for Random Forest: {}".format(rfc.score(X_val, Y_val)))
    print("Testing accuracy for Random Forest: {}".format(rfc.score(X_test, Y_test)))
    joblib.dump(rfc, 'model_files/RF_bow_pac_10.sav')

def RF_tf_idf():
    train = pd.read_csv('data/tf_idf_train.csv')
    test = pd.read_csv('data/tf_idf_test.csv')
    val = pd.read_csv('data/tf_idf_val.csv')
    X_train, Y_train = train[[str(i) for i in range(1000)]].values, train['genre'].values
    X_val, Y_val = val[[str(i) for i in range(1000)]].values, val['genre'].values
    X_test, Y_test = test[[str(i) for i in range(1000)]].values, test['genre'].values
    rfc = RandomForestClassifier(n_estimators=33, criterion='entropy')
    rfc.fit(X_train, Y_train)
    print("Traning accuracy for Random Forest: {}".format(rfc.score(X_train, Y_train)))
    print("Validation accuracy for Random Forest: {}".format(rfc.score(X_val, Y_val)))
    print("Testing accuracy for Random Forest: {}".format(rfc.score(X_test, Y_test)))
    joblib.dump(rfc, 'model_files/RF_tf_idf.sav')

def RF_tf_idf_pca100():
    train = pd.read_csv('data/tf_idf_pca100_train.csv')
    test = pd.read_csv('data/tf_idf_pca100_test.csv')
    val = pd.read_csv('data/tf_idf_pca100_val.csv')
    X_train, Y_train = train[[str(i) for i in range(100)]].values, train['genre'].values
    X_val, Y_val = val[[str(i) for i in range(100)]].values, val['genre'].values
    X_test, Y_test = test[[str(i) for i in range(100)]].values, test['genre'].values
    rfc = RandomForestClassifier(n_estimators=33, criterion='entropy')
    rfc.fit(X_train, Y_train)
    print("Traning accuracy for Random Forest: {}".format(rfc.score(X_train, Y_train)))
    print("Validation accuracy for Random Forest: {}".format(rfc.score(X_val, Y_val)))
    print("Testing accuracy for Random Forest: {}".format(rfc.score(X_test, Y_test)))
    joblib.dump(rfc, 'model_files/RF_tf_idf_pac_100.sav')

def RF_tf_idf_pca10():
    train = pd.read_csv('data/tf_idf_pca10_train.csv')
    test = pd.read_csv('data/tf_idf_pca10_test.csv')
    val = pd.read_csv('data/tf_idf_pca10_val.csv')
    X_train, Y_train = train[[str(i) for i in range(10)]].values, train['genre'].values
    X_val, Y_val = val[[str(i) for i in range(10)]].values, val['genre'].values
    X_test, Y_test = test[[str(i) for i in range(10)]].values, test['genre'].values
    rfc = RandomForestClassifier(n_estimators=33, criterion='entropy')
    rfc.fit(X_train, Y_train)
    print("Traning accuracy for Random Forest: {}".format(rfc.score(X_train, Y_train)))
    print("Validation accuracy for Random Forest: {}".format(rfc.score(X_val, Y_val)))
    print("Testing accuracy for Random Forest: {}".format(rfc.score(X_test, Y_test)))
    joblib.dump(rfc, 'model_files/RF_tf_idf_pac_10.sav')

#RF_bow() #0.603672652192955
#RF_bow_pca100() #0.5706062691609836
#RF_bow_pca10() #0.5562472627166364
#RF_tf_idf() #0.6003566289182256
#RF_tf_idf_pca100() #0.5799599574547957
#RF_tf_idf_pca10() #0.5759557029343678