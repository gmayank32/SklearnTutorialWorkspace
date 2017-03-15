# -*- coding: utf-8 -*-
# above line is set so that if any line of code
# contains non ascii character it should be converted 
# to unicode character

"""Build a language detector model

The goal of this exercise is to train a linear classifier on text features
that represent sequences of up to 3 consecutive characters so as to be
recognize natural languages by using the frequencies of short character
sequences as 'fingerprints'.

"""
# Author: Olivier Grisel <olivier.grisel@ensta.org>
# License: Simplified BSD

import sys 
#for system encoding. Default should be set to utf-8 for printing purposes. 
reload(sys)
sys.setdefaultencoding('utf-8')
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn import metrics


# The training data folder must be passed as first argument
# Data is saved as utf8 encoded characters
languages_data_folder = 'WikipediaAbstracts'
dataset = load_files(languages_data_folder)

# Split the dataset in training and test set:
docs_train, docs_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, test_size=0.5)

# TASK: Build a vectorizer that splits strings into sequence of 1 to 3
# characters instead of word tokens

# TASK: Build a vectorizer / classifier pipeline using the previous analyzer
# the pipeline instance should stored in a variable named clf
clf = Pipeline([('tfidf',TfidfVectorizer(analyzer = 'char'\
					, ngram_range = (1,3), use_idf = True)),
				('sgd', SGDClassifier(alpha = 1e-02, loss = 'hinge'\
					, penalty = 'l2', n_iter = 100))])

# TASK: Fit the pipeline on the training set
clf = clf.fit(docs_train, y_train)

# TASK: Predict the outcome on the testing set in a variable named y_predicted
y_predicted = clf.predict(docs_test)
# Print the classification report
print(metrics.classification_report(y_test, y_predicted,
                                    target_names=dataset.target_names))

# Plot the confusion matrix
cm = metrics.confusion_matrix(y_test, y_predicted)
print(cm)

# import matplotlib.pyplot as plt
# plt.matshow(cm, cmap=plt.cm.jet)
# plt.show()

# Predict the result on some short new sentences:
sentences = [
    u'This is a language detection test.',
    u'এই ভাষা সনাক্তকরণ পরীক্ষা',
    u'Ова е тест за откривање на јазик.',
    u'Ceci est un test de d\xe9tection de la langue.',
    u'Dies ist ein Test, um die Sprache zu erkennen.',
    u'یہ ایک زبان کا پتہ لگانے کے ٹیسٹ ہے.',
    u'هذا هو اختبار الكشف عن اللغة.',
    u'यह एक भाषा पहचान परीक्षण है।'
]

predicted = clf.predict(sentences)
print([dataset.target_names[p] for p in predicted])
for s, p in zip(sentences, predicted):
    print(u'The language of "%s" is "%s"' % (s, dataset.target_names[p]))
