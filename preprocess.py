import spacy
import string
import pandas as pd

def load_data(fname):
    'Reads in a csv file and return a dataframe.'
    return pd.read_csv(fname, nrows = 3000)

def split_data(df):
    '''
    Split data into features with corresponding labels
    Numbers in iloc correspond to column index of csv file
    '''
    features = df.iloc[:, [1, 5, 6, 7, 8]]
    labels = df.iloc[:, [9, 10, 11, 12, 13, 14, 15]]
    return features, labels

def preprocess(df):
    '''
    Preprocess features and returns a word to word location dictionary
    '''
    stopwords = ['a', 'all', 'an', 'and', 'any', 'are', 'as', 'at', 'be', 'been', 'but', 'by', 'few', 'from', 'for', 'have', 'he', 'her', 'here', 'him', 'his', 'how', 'i', 'in', 'is', 'it', 'its', 'many', 'me', 'my', 'none', 'of', 'on', 'or', 'our', 'she', 'some', 'the', 'their', 'them', 'there', 'they', 'that', 'this', 'to', 'us', 'was', 'what', 'when', 'where', 'which', 'who', 'why', 'will', 'with', 'you', 'your', 've', 'm', 't', 's', 'll', 'l']

    word_dict = {}
    nlp = spacy.load("en_core_web_sm")
    for column in df:
        if column != 'company':
            for count, data in enumerate(df[column]):
                tokenized = nlp(str(data).lower())
                for token in tokenized:
                    if token.text not in word_dict and token.text not in stopwords and token.text not in string.digits and token.text not in string.punctuation:
                        word_dict[token.text] = len(word_dict)
    return word_dict

data = load_data('employee_reviews.csv')
features, labels = split_data(data)
print(preprocess(features))
