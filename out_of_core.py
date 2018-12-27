#apply out_of_core learning to make training less computationally expensive

import numpy as np
import re
from nltk.corpus import stopwords
stop = stopwords.words('english')

def tokenizer(text):
	"""cleans the unprocessed text data,
	separates it into word tokens while removing stop words
	"""
	text = re.sub('<[^>]*>', '', text)
	emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
						   text.lower())
	text = re.sub('[\W]+', ' ', text.lower()) \
		   + ' '.join(emoticons).replace('-', '')
	tokenized = [w for w in text.split() if w not in stop]
	return tokenized

def stream_docs(path):
	"""
	reads in and returns one document at a time
	"""
	with open(path, 'r', encoding='utf-8') as csv:
	   next(csv) # skip header
	   for line in csv:
		   text, label = line[:-3], int(line[-2])
		   yield text, label

def get_minibatch(doc_stream, size):
	"""
	take a document stream from the stream_docs function
	return a particular number of documents	specified by the size parameter
	"""
	docs, y = [], []
	try:
		for _ in range(size):
				text, label = next(doc_stream)
				docs.append(text)
				y.append(label)
	except StopIteration:
		return None, None
	return docs, y

#use HasingVectorizer to reduce memory dependence
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
vect = HashingVectorizer(decode_error='ignore',
						 n_features=2**21,
						 preprocessor=None,
						 tokenizer=tokenizer)
clf = SGDClassifier(loss='log', random_state=1, max_iter=1)
doc_stream = stream_docs(path='movie_data.csv')

#start out_of_core learning
import pyprind
pbar = pyprind.ProgBar(45)
classes = np.array([0, 1])
for _ in range(45):
	X_train, y_train = get_minibatch(doc_stream, size=1000)
	if not X_train:
		break
	X_train = vect.transform(X_train)
	clf.partial_fit(X_train, y_train, classes=classes)
	pbar.update()

#eval performance and print accuracy
X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)
print('Accuracy: %.3f' % clf.score(X_test, y_test))

#update model
clf = clf.partial_fit(X_test, y_test)


#Use pickle module to save classifier in its current state, so that we don't need to retrain model when we want to classify new samples
import pickle
import os
dest = os.path.join('movieclassifier', 'pkl_objects')
if not os.path.exists(dest):
    os.makedirs(dest)

pickle.dump(stop,
         open(os.path.join(dest, 'stopwords.pkl'),'wb'),
         protocol=4)
pickle.dump(clf,
         open(os.path.join(dest, 'classifier.pkl'), 'wb'),
         protocol=4)