# Import packages
import pandas as pd
import numpy as np

from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline

from TweetProcess import TweetProcess

# Read training data
train = pd.read_csv( '../data/train.csv' )

# Split the dataset for cross validation
X = train.text
y = train.target
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.25, 
                                                    random_state=123)


# Tf-Idf + Bernoulli NaiveBayes Classfier
from sklearn.feature_extraction.text import TfidfVectorizer
from app.TweetProcess import TweetProcess

# Define Tf-Idf Bernoulli Naive Bayes pipeline
pipe_tfidf_nb = make_pipeline( 
    TweetProcess(),
    TfidfVectorizer(), 
    BernoulliNB() 
    )

# Fit pipeline on train and evaluate on test
nb_fit = pipe_tfidf_nb.fit(X_train, y_train)
print('')
print('Test Classification Report:')
print('')
print( classification_report( y_test, nb_fit.predict(X_test) ) )

# Train model on full dataset
final_model = pipe_tfidf_nb.fit(X, y)

# Pickle final model
import pickle5 as pickle
with open(r"src/app/model.pickle", 'wb') as output_file:
    pickle.dump(final_model, output_file)
