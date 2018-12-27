#Script to construct bag of words model to transform words into feature vectors
import pandas as pd

#read in data
df = pd.read_csv('movie_data.csv', encoding='utf-8')

#First clean text data using python regex library
import re
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text

#apply preprocess function to movie reviews dataframe
df['review'] = df['review'].apply(preprocessor)

#Split text into individual elements
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenizer_porter(text):
   return [porter.stem(word) for word in text.split()]

#remove stopwords from dataset
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')

#train a logistic regression model to classify the movie reviews into positive and negative reviews. 
#divide the DataFrame of cleaned text documents into 25,000 documents for training and 25,000 documents for testing
X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

#use a GridSearchCV object to find the optimal set of parameters for our logistic regression model using 5-fold stratified cross-validation
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)
param_grid = [{'vect__ngram_range': [(1,1)],
              'vect__stop_words': [stop, None],
              'vect__tokenizer': [tokenizer,
                                  tokenizer_porter],
              'clf__penalty': ['l1', 'l2'],
              'clf__C': [1.0, 10.0, 100.0]},
            {'vect__ngram_range': [(1,1)],
              'vect__stop_words': [stop, None],
              'vect__tokenizer': [tokenizer,
                                  tokenizer_porter],
              'vect__use_idf':[False],
              'vect__norm':[None],
              'clf__penalty': ['l1', 'l2'],
              'clf__C': [1.0, 10.0, 100.0]}
            ]
lr_tfidf = Pipeline([('vect', tfidf),
                    ('clf',
                     LogisticRegression(random_state=0))])
gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                          scoring='accuracy',
                          cv=5, verbose=1,
                          n_jobs=-1)
gs_lr_tfidf.fit(X_train, y_train)

#print best parameter set
print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
#print the average 5-fold cross-validation accuracy scores on the training set and the classification accuracy on the test dataset, using best model
print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)
clf = gs_lr_tfidf.best_estimator_
print('Test Accuracy: %.3f' % clf.score(X_test, y_test))