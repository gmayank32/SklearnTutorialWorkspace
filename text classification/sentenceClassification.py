from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

text_clf = Pipeline([('vect', CountVectorizer()),
					('tfidf', TfidfTransformer()),
					('clf', SGDClassifier(loss='hinge', penalty='l2',
						n_iter=100))
					])

data = pd.read_csv('labelledData.csv')

train_data = data.sample(frac=.8,random_state=31)
test_data = data.sample(frac=.2,random_state=41)

parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
               'tfidf__use_idf': (True, False),
               'clf__alpha': (1e-2, 1e-3),
}

text_clf = text_clf.fit(train_data.Sentence, train_data.Label)
gs_clf = GridSearchCV(text_clf, parameters, n_jobs = -1)
gs_clf = gs_clf.fit(train_data.Sentence, train_data.Label)
print gs_clf.best_score_

for param_name in sorted(parameters.keys()):
    print param_name, gs_clf.best_params_[param_name]

# predicted = text_clf.predict(test_data.Sentence)

# print np.mean(predicted == test_data.Label)

