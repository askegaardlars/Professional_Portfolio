# Computed on MacOS 12.0.1 and Windows 11

# sklearn is our chosen library for machine learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.metrics import accuracy_score,hamming_loss
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

import statistics

import ast

import pandas as pd

import numpy as np

import random

import sys

import warnings

# Warnings are ignored in order to streamline output. 
warnings.filterwarnings("ignore")


# Using Newton algorithm to solve logistic regression
lr = LogisticRegression(solver='newton-cg')

# iter_test a fuction that automates running n number of logistic regression tests
# line *** defines top5, a string array of 81 indices that hold the top 5 problem code predictions as strings in rank order
# line *** defines correct, a string array of 81 indices that hold the correctly predicted problem codes as strings in rank order
# line *** defines actual, a string array of 81 indices that hold the correct problem codes that were not included predictions
# line *** 
def iter_test(path, n, clf):
    df = pd.read_csv(r'{path}')
    # Read in data set from personal computer


    # mlb stores shortened name for MultiLabelBinarizer
    mlb = MultiLabelBinarizer()

    # Separates and binarizes device problem according to the key '; '
    DP = mlb.fit_transform(df["Device Problem"].str.split("; "))
    
    top5 = []
    scoring_result = []
    correct = []
    missing = []
    actual = []
    index = 0
    for i in range(n):
        top5.append([])
        scoring_result.append([])
        correct.append([])
        missing.append([])
        actual.append([])
        tfidf = TfidfVectorizer(analyzer='word', max_features=10000, ngram_range=(1,3), stop_words='english')
        X = tfidf.fit_transform(df['Event Text']).toarray()
        X_train, X_test, DP_train, DP_test = train_test_split(X, DP, test_size=0.05, random_state=random.randint(0,42))
        clf.fit(X_train, DP_train)
        DP_pred = clf.predict(X_test)
        top5[index], scoring_result[index], correct[index], missing[index], actual[index] = scoring(clf, X_test, DP_test, DP_pred)
        index = index + 1
    return top5, scoring_result, correct, missing, actual


def scoring(clf, X_test, DP_test, DP_pred):
    index1 = 0
    index2 = 0
    top5 = get_five(clf, X_test)
    length = []
##    top3 = get_three(clf, X_test)
    actual = []
    while index1 < len(DP_test):
        actual.append([])
        index2 = 0
        for i in DP_test[index1]:
            if i == 1:
                actual[index1].append(mlb.classes_[index2])
            index2 = index2 + 1
        index1 = index1 + 1
    for i in actual:
        length.append(len(i))
    scoring_result = []
    missing = []
    correct = []
    index = 0
    while index < len(actual):
        count = 0
        status = True
        missing.append([])
        correct.append([])
        for j in actual[index]:
            for k in top5[index]:
                if j == k:
                    count = count + 1
                    status = True
                    break
                else:
                    status = False
            if status == False:
                missing[index].append(j)
        correct[index] = [x for x in actual[index] if x not in missing[index]]
        scoring_result.append(count/len(actual[index]))
        index = index + 1
    temp = max(length) + 1
    n_count_true = [0] * temp
    n_count_total = [0] * temp
    for i in range(len(scoring_result)):
        if scoring_result[i] == 1:
            for j in range(max(length)+1):
                if j == length[i]:
                    n_count_true[j] = n_count_true[j] + 1
                    n_count_total[j] = n_count_total[j] + 1
        else:
            for j in range(max(length)+1):
                if j == length[i]:
                    n_count_total[j] = n_count_total[j] + 1
    correct_count = 0
    print("Classifier: {}".format(clf))
    for i in range(len(scoring_result)):
        if scoring_result[i] == 1:
            print(i + 1,". All the actual labels are in the top 5 predicted labels")
            print("The actual label are ", actual[i])
            print("The top 5 labels prediction are ",top5[i])
            correct_count = correct_count + 1
        else:
            print(i+1,". {}% of the actual labels are in the top 5 predicted labels".format(scoring_result[i]*100,'%'))
            print("The correct predicted label(s) is(are) ", correct[i])
            print("The missing label is(are) ", missing[i])
            print("The top 5 predicted label are ", top5[i])
    print()
    for i in range(1, max(length)+1):
        print("{} actual label completely predicted correctly: {}/{}".format(i, n_count_true[i], n_count_total[i]))
    print("The number of completely correct output are ", correct_count)
    print("The average precision for {} instances are {}".format(len(actual),average(scoring_result)*100))
    print("The median is ", statistics.median(scoring_result)*100)
    print("---------------------------------------------")
    return top5, scoring_result, correct, missing, actual

def average(lst):
    return sum(lst)/len(lst)

# Optional function that self-defines and implements Jaccard Scoring
# def j_score(DP_true, DP_pred):
#     jaccard = np.minimum(DP_true, DP_pred).sum(axis = 1)/np.maximum(DP_true, DP_test).sum(axis = 1)
#     return jaccard.mean()*100

# data_frame_generate is a function that creates a dataframe with 2 columns: actual device problem (index 0) and predicted device problem (index 1)   
def data_frame_generate(DP_test, DP_pred):
    d = {'Actual':DP_test,'Predict':DP_pred}
    df = pd.DataFrame(d)
    return df

# get_five is a function that retrieves the 5 highest-confidence device problem predictions for each instance in test data set
def get_five(clf, X_test):
    probs = clf.predict_proba(X_test)
    best5 = np.argsort(-probs, axis=1)[:,:5]
    index1 = 0
    result_list = []
    for i in best5:
        result_list.append([])
        for j in i:
            result_list[index1].append(mlb.classes_[j])
        index1 = index1 + 1
    return result_list

# print_result is a function that prints scoring metrics to computer terminal
def print_result(DP_pred, DP_test, clf):
    print("Classifier: ", clf.__class__.__name__)
    print("Jacard score: {}".format(jaccard_score(DP_test, DP_pred, average='samples')))
    print("F1 score micro: {}".format(f1_score(DP_test, DP_pred, average='micro')))
    print("F1 score macro: {}".format(f1_score(DP_test, DP_pred, average='macro')))
    print("F1 score weighted: {}".format(f1_score(DP_test, DP_pred, average='weighted')))
    print("F1 score samples: {}".format(f1_score(DP_test, DP_pred, average='samples')))
    df = data_frame_generate(mlb.inverse_transform(DP_test), mlb.inverse_transform(DP_pred))
    with pd.option_context('display.max_rows', None,
                    'display.precision', 3,
                    ):print(df)
    df.to_excel('{}.xlsx'.format(clf),index=False)
    print("---------------------------")

# mlb stores shortened name for MultiLabelBinarizer
mlb = MultiLabelBinarizer()

def dff_test_set(path_to_train, path_to_test, n_max_feature, clf):
    train_df = pd.read_csv(path_to_train)
    test_df = pd.read_csv(path_to_test)
    top5 = []
    scoring_result = []
    correct = []
    missing = []
    actual = []
    tfidf = TfidfVectorizer(analyzer='word', max_features=n_max_feature, ngram_range=(1,3), stop_words='english')
    DP_train = mlb.fit_transform(train_df["Device Problem"].str.split("; "))
    X_train = tfidf.fit_transform(train_df['Event Text']).toarray()
    X_test = tfidf.transform(test_df['Event Text']).toarray()
    DP_test = test_df['Device Problem']
    clf.fit(X_train, DP_train)
    DP_pred = clf.predict(X_test)
    top5, scoring_result, correct, missing, actual = scoring_dff_test(clf, X_test, DP_test, DP_pred)
    return top5, scoring_result, correct, missing, actual

def scoring_dff_test(clf, X_test, DP_test, DP_pred):
    index1 = 0
    index2 = 0
    top5 = get_five(clf, X_test)
    length = []
##    top3 = get_three(clf, X_test)
    actual = []
    for i in range(len(DP_test)):
        actual.append(DP_test[i].split("; "))
    for i in actual:
        length.append(len(i))
    scoring_result = []
    missing = []
    correct = []
    index = 0
    while index < len(actual):
        count = 0
        status = True
        missing.append([])
        correct.append([])
        for j in actual[index]:
            for k in top5[index]:
                if j == k:
                    count = count + 1
                    status = True
                    break
                else:
                    status = False
            if status == False:
                missing[index].append(j)
        correct[index] = [x for x in actual[index] if x not in missing[index]]
        scoring_result.append(count/len(actual[index]))
        index = index + 1
    temp = max(length) + 1
    n_count_true = [0] * temp
    n_count_total = [0] * temp
    for i in range(len(scoring_result)):
        if scoring_result[i] == 1:
            for j in range(max(length)+1):
                if j == length[i]:
                    n_count_true[j] = n_count_true[j] + 1
                    n_count_total[j] = n_count_total[j] + 1
        else:
            for j in range(max(length)+1):
                if j == length[i]:
                    n_count_total[j] = n_count_total[j] + 1
    correct_count = 0
    print("Classifier: {}".format(clf))
    for i in range(len(scoring_result)):
        if scoring_result[i] == 1:
            print(i + 1,". All the actual labels are in the top 5 predicted labels")
            print("The actual label are ", actual[i])
            print("The top 5 labels prediction are ",top5[i])
            correct_count = correct_count + 1
        else:
            print(i+1,". {}% of the actual labels are in the top 5 predicted labels".format(scoring_result[i]*100,'%'))
            print("The correct predicted label(s) is(are) ", correct[i])
            print("The missing label is(are) ", missing[i])
            print("The top 5 predicted label are ", top5[i])
    print()
    for i in range(1, max(length)+1):
        print("{} actual label completely predicted correctly: {}/{}".format(i, n_count_true[i], n_count_total[i]))
    print("The number of completely correct output are ", correct_count)
    print("The average precision for {} instances are {}".format(len(actual),average(scoring_result)*100))
    print("The median is ", statistics.median(scoring_result)*100)
    print("---------------------------------------------")
    return top5, scoring_result, correct, missing, actual

##train_df = pd.read_csv(r'C:\Users\huynh\OneDrive\Desktop\Practicum\train.csv')
##test_df = pd.read_csv(r'C:\Users\huynh\OneDrive\Desktop\Practicum\test.csv')
##tfidf = TfidfVectorizer(analyzer='word', max_features=10000, ngram_range=(1,3), stop_words='english')
##print(type(train_df["Device Problem"]))
##print(type(test_df["Device Problem"]))
##DP_train = mlb.fit_transform(train_df["Device Problem"].str.split("; "))
##X_train = tfidf.fit_transform(train_df['Event Text']).toarray()
##X_test = tfidf.fit_transform(test_df['Event Text']).toarray()
##DP_test = test_df['Device Problem']
##actual = []
##for i in range(len(DP_test)):
##    actual.append(DP_test[i].split("; "))
clf = OneVsRestClassifier(lr)
top5, scoring_result, correct, missing, actual = dff_test_set(r'C:\Users\huynh\OneDrive\Desktop\Practicum\train.csv', r'C:\Users\huynh\OneDrive\Desktop\Practicum\test.csv', 9000, clf) 

# clf stores shortened name for OneVsRestClassifier(lr) function, where argument lr denotes the type logistic regresssion
clf = OneVsRestClassifier(lr)

##path = 'C:\Users\huynh\OneDrive\Desktop\PracticumlematizedData.csv'
# iter_test(path, 3, clf) assigns return values in order to: top5, scoring_result, correct, missing, actual 
##top5, scoring_result, correct, missing, actual = iter_test(df, 3, clf)
