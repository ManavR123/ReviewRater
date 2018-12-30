#Update the predictive model from the feedback data that is being collected in the SQLite database.

import pickle
import sqlite3
import numpy as np
import os

# import HashingVectorizer from local dir
from vectorizer import vect


cur_dir = os.path.dirname(__file__)

clf = pickle.load(open(os.path.join(cur_dir,
                  'pkl_objects',
                  'classifier.pkl'), 'rb'))
db = os.path.join(cur_dir, 'reviews.sqlite')

conn = sqlite3.connect(db)
c = conn.cursor()
c.execute('SELECT * from review_db')

results = c.fetchmany(5000)
data = np.array(results)
X = data[:, 0]
y = data[:, 1].astype(int)
X_train = vect.transform(X)
print('Accuracy: %.3f' % clf.score(X_train, y))
conn.close()