import pandas as pd

def baseline(train_set):
    maximal_freq_genre = train_set['genre'].value_counts().idxmax()
    return maximal_freq_genre

def eval_baseline(test_set, Y_baseline):
    a = test_set['genre'].value_counts()[Y_baseline]
    b = test_set['genre'].value_counts().sum()
    return (a / b) * 100

train_set = pd.read_csv('data/bow_train.csv')
test_set = pd.read_csv('data/bow_test.csv')
Y_baseline = baseline(train_set)
print("Performance of Baseline: {:} ".format(eval_baseline(test_set, Y_baseline)))