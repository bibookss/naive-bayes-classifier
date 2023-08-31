import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet, words
from nltk import pos_tag, download
from collections import Counter
from nltk.metrics.distance import jaccard_distance
from nltk.util import ngrams

class NaiveBayesSpamClassifier:
    def __init__(self, smoothing=1):
        download('stopwords', quiet=True)
        download('punkt', quiet=True)
        download('wordnet', quiet=True)
        download('averaged_perceptron_tagger', quiet=True)
        download('words', quiet=True)

        self.smoothing = smoothing
        self.vocab = {}
        self.prior = {}
        self.likelihood = {}
        self.classes = ['spam', 'ham']
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english'))
        self.correct_words = words.words()
        
    def penn_to_wordnet_pos(self, tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN
        
    def preprocess(self, X):
        X = X.str.lower()
        X = X.str.replace(r'[^a-zA-Z\s]', '', regex=True)
        X = X.apply(word_tokenize)
        X = X.apply(pos_tag)
        X = X.apply(lambda x: [(word, tag) for word, tag in x if word not in self.stopwords])
        X = X.apply(lambda x: [self.lemmatizer.lemmatize(word, pos=self.penn_to_wordnet_pos(tag)) for word, tag in x])
        return X

    def fit(self, X, y):
        X = self.preprocess(X)

        for _class in self.classes:
            word_counter = Counter()
            for i in range(len(X)):
                if y[i] == _class:
                    word_counter.update(X[i])
            self.vocab[_class] = word_counter

            # print('After preprocessing, class {} has {} unique words with {} total words'.format(_class, len(word_counter), sum(word_counter.values())))
            # print('10 most common words: {}'.format(word_counter.most_common(10)))

        # Calculate total words
        spam_words = sum(self.vocab['spam'].values())
        spam_unique_words = len(self.vocab['spam'])
        ham_words = sum(self.vocab['ham'].values())
        ham_unique_words = len(self.vocab['ham'])
        total_words = spam_words + ham_words 
        # print('Total words in spam and ham is {}'.format(total_words))

        # Calculate prior
        self.prior['spam'] = spam_words / total_words
        self.prior['ham'] = ham_words / total_words
        # print('Prior for spam is {}'.format(self.prior['spam']))
        # print('Prior for ham is {}'.format(self.prior['ham']))

        # Calculate likelihood
        self.likelihood['spam'] = {}
        for word in self.vocab['spam']:
            self.likelihood['spam'][word] =  (self.vocab['spam'][word] + self.smoothing) / (spam_words + self.smoothing * spam_unique_words)
        # Add one entry for unknown word
        self.likelihood['spam']['UNKOWN_WORD'] = self.smoothing / (spam_words + self.smoothing * spam_unique_words)

        self.likelihood['ham'] = {}
        for word in self.vocab['ham']:
            self.likelihood['ham'][word] =  self.vocab['ham'][word] / (ham_words + self.smoothing * spam_unique_words)
        # Add one entry for unknown word
        self.likelihood['ham']['UNKOWN_WORD'] = self.smoothing / (ham_words + self.smoothing * spam_unique_words)

        # Compute marginal probability
        self.marginal = {}
        for _class in self.classes:
            for word in self.vocab[_class]:
                if word in self.marginal:
                    self.marginal[word] += self.vocab[_class][word]
                else:
                    self.marginal[word] = self.vocab[_class][word]
        
        for word in self.marginal:
            self.marginal[word] = self.marginal[word] / (total_words + self.smoothing * (spam_unique_words + ham_unique_words))

        # Add one entry for unknown word
        self.marginal['UNKOWN_WORD'] = self.smoothing / (total_words + self.smoothing * (spam_unique_words + ham_unique_words))

    def predict(self, X):
        X = self.preprocess(X)
        y_pred = []
        for i in range(len(X)):
            sentence = X[i]
            ham = 1
            spam = 1
            for word in sentence:
                if word in self.likelihood['ham']:
                    ham *= (self.likelihood['ham'][word] * self.prior['ham'] / self.marginal[word])
                else:
                    ham *= (self.likelihood['ham']['UNKOWN_WORD'] * self.prior['ham'] / self.marginal['UNKOWN_WORD'])

                if word in self.likelihood['spam']:
                    spam *= (self.likelihood['spam'][word] * self.prior['spam'] / self.marginal[word])
                else:
                    spam *= (self.likelihood['spam']['UNKOWN_WORD'] * self.prior['spam'] / self.marginal['UNKOWN_WORD'])

            if ham >= spam:
                y_pred.append('ham')
            else:
                y_pred.append('spam')
        
        return y_pred

    def score(self, X, y):
        pass