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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from preprocess import load_data, split_data, preprocess
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

def main(in_file):
    df = load_data(in_file)
    features, Y_train, Y_test = split_data(df)
    X_train, X_test = preprocess(features)
    X_train.to_csv('X_train.csv')
    X_test.to_csv('X_test.csv')
    Y_train.to_csv('Y_train.csv')
    Y_test.to_csv('Y_test.csv')
    X_train = pd.read_csv('X_train.csv')
    X_test = pd.read_csv('X_test.csv')
    Y_train = pd.read_csv('Y_train.csv')
    Y_test = pd.read_csv('Y_test.csv')

    cosineSim(X_train)
    classProbs, condProbs, vocabSize, proVocabSize, conVocabSize = trainSentimentAnalysis(X_train)

    accuracy = 0
    for review in X_test['pros']:
        result = testSentimentAnalysis(str(review).split(), classProbs, condProbs, vocabSize, proVocabSize, conVocabSize)

        if result == 'pro':
            accuracy += 1
    for review in X_test['cons']:
        result = testSentimentAnalysis(str(review).split(), classProbs, condProbs, vocabSize, proVocabSize, conVocabSize)

        if result == 'con':
            accuracy += 1
    accuracy = float(accuracy / (len(X_test['pros']) + len(X_test['cons'])))
    print('Accuracy for Naive Bayes Sentiment Analysis:', accuracy)

    trainModels(X_train, X_test, Y_train, Y_test)
    return 'done'
def trainModels(X_train, X_test, Y_train, Y_test):
    xcolumns = ['summary', 'pros', 'cons', 'advice-to-mgmt']
    ycolumns = ['company', 'overall-ratings', 'work-balance-stars', 'culture-values-stars','carrer-opportunities-stars','comp-benefit-stars','senior-mangemnet-stars','helpful-count']

    for xcolumn in xcolumns:
        print('Using feature', xcolumn + ':')
        train, test, vectorizer = generateFeatureMatrix(X_train, X_test, xcolumn)
        for ycolumn in ycolumns:
            print('Passive Aggressive accuracy for', ycolumn + ':', passiveAggresive(train, test, Y_train, Y_test, ycolumn))
            print('Naive Bayes Accuracy for', ycolumn + ':', naiveBayes(vectorizer, train, test, Y_train, Y_test, ycolumn))
            print('Support Vector Machine accuracy for', ycolumn + ':', SVM(train, test, Y_train, Y_test, ycolumn))
            print('Decision Tree accuracy for', ycolumn + ':', decisionTree(train, test, Y_train, Y_test, ycolumn))
            print('Neural Network accuracy for', ycolumn + ':', neuralNetwork(train, test, Y_train, Y_test, ycolumn))
            print('Random Forest accuracy for', ycolumn + ':', randomForest(train, test, Y_train, Y_test, ycolumn))
            print('AdaBoost accuracy for', ycolumn + ':', adaBoost(train, test, Y_train, Y_test, ycolumn))
    return 'done'
def cosineSim(X_train):
    vectorizer = CountVectorizer()
    train = vectorizer.fit_transform(X_train['summary'].values.astype('U'))
    sim = []
    for i in train:
        for j in train:
            sim.append(cosine_similarity(i, j))
    return sim
def trainSentimentAnalysis(X_train):
    '''
    Predict whether given text is a Pro or a Con
    '''
    proReviews = [str(review).split() for review in X_train['pros']]
    conReviews = [str(review).split() for review in X_train['cons']]

    classProbs = {}
    classProbs['pros'] = len(proReviews) / (len(proReviews) + len(conReviews))
    classProbs['cons'] = len(conReviews) / (len(proReviews) + len(conReviews))

    'Create Vocabulary to determine Vocabulary Size'
    vocab = {}
    vocabSize = 0
    for review in proReviews:
        for word in review:
            if word not in vocab.keys() and word:
                vocab[word] = 1
                vocabSize += 1
            elif word in vocab.keys():
                vocab[word] += 1
    proVocabSize = vocabSize

    for review in conReviews:
        for word in review:
            if word not in vocab.keys():
                vocab[word] = 1
                vocabSize += 1
            elif word in vocab.keys():
                vocab[word] += 1
    conVocabSize = vocabSize - proVocabSize

    proProbs = {}
    conProbs = {}

    'Calculate conditional word probabilities for pro reviews'
    for review in proReviews:
        for word in review:
            if word not in proProbs.keys():
                proProbs[word] = 1 / float(proVocabSize + vocabSize)
            elif word in vocab.keys():
                proProbs[word] += 1 / float(proVocabSize + vocabSize)

    'Calculate conditional word probabilities for con reviews'
    for review in conReviews:
        for word in review:
            if word not in conProbs.keys():
                conProbs[word] = 1 / float(conVocabSize + vocabSize)
            elif word in vocab.keys():
                conProbs[word] += 1 / float(conVocabSize + vocabSize)

    condProbs = [proProbs, conProbs]
    return classProbs, condProbs, vocabSize, proVocabSize, conVocabSize

def testSentimentAnalysis(review, classProbs, condProbs, vocabSize, proVocabSize, conVocabSize):

    proProb = np.log(classProbs['pros'])
    proIndex = 0
    for word in review:
        if word not in condProbs[proIndex].keys():
            condProbs[proIndex][word] = 1 / float(proVocabSize + vocabSize)
        proProb += np.log(condProbs[proIndex][word])
    conProb = np.log(classProbs['cons'])
    conIndex = 1
    for word in review:
        if word not in condProbs[conIndex].keys():
            condProbs[conIndex][word] = 1 / float(conVocabSize + vocabSize)
        conProb += np.log(condProbs[conIndex][word])

    if proProb > conProb:
        return 'pro'
    else:
        return 'con'

def generateFeatureMatrix(X_train, X_test, column, vectorizer = CountVectorizer()):
    '''
    Convert textual training and testing data into integer matrices
    '''
    train = vectorizer.fit_transform(X_train[column].values.astype('U'))
    test = vectorizer.transform(X_test[column].values.astype('U'))
    return train, test, vectorizer

def companyConverter(companyInt):
    '''
    Convert Company integer to string
    '''
    if companyInt == 0:
        return 'Amazon'
    elif companyInt == 1:
        return 'Apple'
    elif companyInt == 2:
        return 'Facebook'
    elif companyInt == 3:
        return 'Google'
    elif companyInt == 4:
        return 'Microsoft'
    elif companyInt == 5:
        return 'Netflix'

def passiveAggresive(train, test, Y_train, Y_test, column):
    '''
    Fits a Passive Aggresive Perceptron Classifer
    '''
    clf = PassiveAggressiveClassifier(C = .1, max_iter = 1000, class_weight = 'balanced', tol = 1e-3)
    clf.fit(train, Y_train[column])
    clf.predict(test)
    return clf.score(test, Y_test[column])

def naiveBayes(vectorizer, train, test, Y_train, Y_test, column):
    '''
    Fits a Multinomial Naive Bayes Classifer and finds the top 10 important words assocaited with a feature
    '''
    clf = MultinomialNB()
    clf.fit(train, Y_train[column])
    clf.predict(test)

    if column == 'company':
        feature_names = vectorizer.get_feature_names()
        for count, company in enumerate(clf.feature_log_prob_):
            arr = (-company).argsort()[:10]
            print('Top 10 words associated with', companyConverter(count))
            for i in arr:
                print(feature_names[int(i)])

    return clf.score(test, Y_test[column])

def SVM(train, test, Y_train, Y_test, column):
    '''
    Fits a Support Vector Machine Classifer
    '''
    scores = []
    clf = SVC(C = .5, kernel = 'linear', gamma = 0.001)
    folds = StratifiedKFold(n_splits = 5, shuffle = False)
    for training, testing in folds.split(train, Y_train[column]):
        X_train = train[training]
        X_test = train[testing]
        y_train = Y_train[column][training]
        y_test = Y_train[column][testing]

        clf.fit(X_train, y_train)

        clf.predict(X_test)
        scores.append(clf.score(X_test, y_test))

    return np.array(scores).mean()

def decisionTree(train, test, Y_train, Y_test, column):
    '''
    Fits a Decision Tree Classifer
    '''
    clf = tree.DecisionTreeClassifier(criterion = 'entropy')
    clf.fit(train, Y_train[column])
    clf.predict(test)
    return clf.score(test, Y_test[column])

def neuralNetwork(train, test, Y_train, Y_test, column):
    '''
    Fits a MultiLayer Perceptron Neural Network
    '''
    clf = MLPClassifier(solver = 'adam', activation = 'relu', max_iter = 100)
    clf.fit(train, Y_train[column])
    clf.predict(test)
    return clf.score(test, Y_test[column])

def randomForest(train, test, Y_train, Y_test, column):
    '''
    Custom implmentation for Random Forest
    '''

    'Creates a Decision Tree forest of size 40 '
    clfs = [tree.DecisionTreeClassifier(criterion = 'entropy', ) for i in range(1, 41)]

    'For each Decision Tree, fit a classifer to the bootstrapped data'
    for clf in clfs:
        X, y = utils.resample(train, Y_train[column], replace = True)
        clf.fit(X, y)

    'Get predicitons for each classifier'
    predictions = []
    for clf in clfs:
        predictions.append(clf.predict(test))

    'Classify by majority vote'
    y_pred = []
    for prediction in np.array(predictions).T:
        count = Counter(prediction).most_common()
        count = [tuple for tuple, num in count if num == count[0][1]]
        y_pred.append(random.choice(count))
    return metrics.accuracy_score(Y_test[column], np.array(y_pred))

def adaBoost(train, test, Y_train, Y_test, column, M = 11):
    '''
    Runs Adaptive Boosting Algorithm
    '''
    clf = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth = 1))
    clf.fit(train, Y_train[column])
    clf.predict(test)
    return clf.score(test, Y_test[column])
if __name__ == '__main__':
	main(sys.argv[1])
