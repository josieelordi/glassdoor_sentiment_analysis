# EECS 486 - Final Project/classifer.py
# Pranav Ajith, Daniel Chandross, Tyler Eastman, Josie Elordi, Rana Makki

import os
import re
import sys
import random
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.svm import SVC
from collections import Counter
from sklearn import metrics, utils
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import load_data, split_data, preprocess

def main(in_file):
    # df = load_data(in_file)
    # features, Y_train, Y_test = split_data(df)
    # X_train, X_test = preprocess(features)
    # X_train.to_csv('X_train.csv')
    # X_test.to_csv('X_test.csv')
    # Y_train.to_csv('Y_train.csv')
    # Y_test.to_csv('Y_test.csv')
    X_train = pd.read_csv('X_train.csv')
    X_test = pd.read_csv('X_test.csv')
    Y_train = pd.read_csv('Y_train.csv')
    Y_test = pd.read_csv('Y_test.csv')

    vectorizer = TfidfVectorizer()
    train = vectorizer.fit_transform(X_train['summary'].values.astype('U'))
    test = vectorizer.transform(X_test['summary'].values.astype('U'))


    print('Naive Bayes Accuracy:', naiveBayes(train, test, Y_train, Y_test))
    print('Suppor Vector Machine Accuracy:', SVM(train, test, Y_train, Y_test))
    print('Decision Tree Accuracy:', decisionTree(train, test, Y_train, Y_test))
    print('Neural Network Accuracy:', neuralNetwork(train, test, Y_train, Y_test))
    print('Random Forest Accuracy:', randomForest(train, test, Y_train, Y_test))
def naiveBayes(train, test, Y_train, Y_test):
    clf = MultinomialNB()
    clf.fit(train, Y_train['company'])
    clf.predict(test)
    return clf.score(test, Y_test['company'])
def SVM(train, test, Y_train, Y_test):
    clf = SVC(C = 0.55, kernel = 'linear', gamma = 'auto')
    clf.fit(train, Y_train['company'])
    clf.predict(test)
    return clf.score(test, Y_test['company'])
def decisionTree(train, test, Y_train, Y_test):
    clf = tree.DecisionTreeClassifier(criterion = 'entropy')
    clf.fit(train, Y_train['company'])
    clf.predict(test)
    return clf.score(test, Y_test['company'])
def neuralNetwork(train, test, Y_train, Y_test):
    clf = MLPClassifier(solver = 'adam', activation = 'relu', max_iter = 100)
    clf.fit(train, Y_train['company'])
    clf.predict(test)
    return clf.score(test, Y_test['company'])
def randomForest(train, test, Y_train, Y_test):
    clfs = [tree.DecisionTreeClassifier(criterion = 'entropy') for i in range(1, 21)]
    for clf in clfs:
        X, y = utils.resample(train, Y_train['company'], replace = True)
        clf.fit(X, y)
    predictions = []
    for clf in clfs:
        predictions.append(clf.predict(test))
    y_pred = []
    for prediction in np.array(predictions).T:
        count = Counter(prediction).most_common()
        count = [tuple for tuple, num in count if num == count[0][1]]
        y_pred.append(random.choice(count))
    return metrics.accuracy_score(Y_test['company'], np.array(y_pred))
if __name__ == '__main__':
	main(sys.argv[1])
