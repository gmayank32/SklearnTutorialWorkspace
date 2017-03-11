"""Build a sentiment analysis / polarity model

Sentiment analysis can be casted as a binary text classification problem,
that is fitting a linear classifier on features extracted from the text
of the user messages so as to guess wether the opinion of the author is
positive or negative.

In this examples we will use a movie review dataset.

"""
# Author: Olivier Grisel <olivier.grisel@ensta.org>
# License: Simplified BSD

import sys
import numpy as np
from os.path import isdir,join
from os import listdir
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
import pandas as pd
from os import *


if __name__ == "__main__":
    # NOTE: we put the following in a 'if __name__ == "__main__"' protected
    # block to be able to use a multi-core grid search that also works under
    # Windows, see: http://docs.python.org/library/multiprocessing.html#windows
    # The multiprocessing module is used as the backend of joblib.Parallel
    # that is used when n_jobs != 1 in GridSearchCV

    # the training data folder must be passed as first argument
    movie_reviews_data_folder = 'moviesreview'
    dataset = load_files(movie_reviews_data_folder, shuffle=True, \
                        load_content=True, encoding = "UTF-8")
    print("n_samples: %d" % len(dataset.data))

    # split the dataset in training and test set:
    docs_train, docs_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.25, random_state=None)
    # TASK: Build a vectorizer / classifier pipeline that filters out tokens
    # that are too rare or too frequent
    linear_model = ('lsvc', LinearSVC(penalty='l2', loss='hinge',random_state=41, max_iter=50))
    SGDClassifier_model = ('sgdc', SGDClassifier(penalty='l2', loss='hinge',random_state=41, n_iter=100))
    clf = Pipeline([
                    ('tfidf', TfidfVectorizer()),
                    SGDClassifier_model
                    ])
    text_clf = clf.fit(docs_train, y_train)
    # TASK: Build a grid search to find out whether unigrams or bigrams are
    # more useful.
    # Fit the pipeline on the training set using grid search for the parameters
    parameters = {  'tfidf__ngram_range':[(1, 1), (1,2)],
                    'tfidf__use_idf':[True, False],
                    'sgdc__alpha': [1e-03, 1e-02, 1e-01, 1, 10, 1e02]
                    #'lsvc__C': (1e-03, 1e-01, 1, 10, 1e02)
                }
    gs_clf = GridSearchCV(clf, parameters, n_jobs = -1, verbose = 100000)
    gs_clf = gs_clf.fit(docs_train, y_train)
    # TASK: print the cross-validated scores for the each parameters set
    # explored by the grid search
    print gs_clf.best_score_
    for param_name in sorted(parameters.keys()):
        print(param_name, gs_clf.best_params_[param_name])
    # TASK: Predict the outcome on the testing set and store it in a variable
    # named y_predicted
    y_predicted = gs_clf.predict(docs_test)
    # Print the classification report
    print(metrics.classification_report(y_test, y_predicted, 
                                        target_names=dataset.target_names))
    # Print and plot the confusion matrix
    cm = metrics.confusion_matrix(y_test, y_predicted)
    print(cm)
    import matplotlib.pyplot as plt
    plt.matshow(cm)
    plt.show()
