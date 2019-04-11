# EECS 486 - Final Project/preprocess.py
# Pranav Ajith, Daniel Chandross, Tyler Eastman, Josie Elordi, Rana Makki
import spacy
import string
import pandas as pd

def load_data(fname):
    'Reads in a csv file and return a dataframe'
    return pd.read_csv(fname, nrows = 10000)

def split_data(df):
    '''
    Split data into features and corresponding labels
    Numbers in iloc correspond to column index of csv file
    '''
    features = df.iloc[:, [1, 5, 6, 7, 8]]
    labels = df.iloc[:, [9, 10, 11, 12, 13, 14, 15]]
    return features, labels[:8000], labels[8000:]

def preprocess(df):
    '''
    Preprocess features and tokenize dataframe
    '''

    stopwords = ['a', 'all', 'an', 'and', 'any', 'are', 'as', 'at', 'be', 'been', 'but', 'by', 'few', 'from', 'for', 'have', 'he', 'her', 'here', 'him', 'his', 'how', 'i', 'in', 'is', 'it', 'its', 'many', 'me', 'my', 'none', 'of', 'on', 'or', 'our', 'she', 'some', 'the', 'their', 'them', 'there', 'they', 'that', 'this', 'to', 'us', 'was', 'what', 'when', 'where', 'which', 'who', 'why', 'will', 'with', 'you', 'your', 've', 'm', 't', 's', 'll', 'l']

    dataDict = {'company': [], 'summary': [], 'pros': [], 'cons': [], 'advice-to-mgmt': []}

    'Fill data dictionary with tokenized Glassdoor data'
    nlp = spacy.load("en_core_web_sm")
    for column in df:
        if column == 'company':
            dataDict[column] = [data for data in df[column]]
        elif column != 'company':
            for count, data in enumerate(df[column]):
                tokenized = nlp(str(data).lower())
                temp = []
                for token in tokenized:
                    if token.text not in stopwords and token.text not in string.digits and token.text not in string.punctuation:
                        temp.append(token.text)
                dataDict[column].append(' '.join(temp))
    features = pd.DataFrame.from_dict(dataDict)
    return features[:8000], features[8000:]
