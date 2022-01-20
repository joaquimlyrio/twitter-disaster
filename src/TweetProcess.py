import pandas as pd
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import string
import re
from nltk.stem import PorterStemmer
nltk.download('stopwords')
nltk.download('punkt')

from sklearn.base import BaseEstimator, TransformerMixin

##
## Class that cleans tweet text
##
class TweetProcess(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.step = 'TweetProcess'

    def fit(self, X, y=None):
       return self

    def transform(self, X, y=None):
        return X.apply(lambda row: self.transform_one_tweet(row))

    def transform_one_tweet(self, tweet):

        # Get stopwords
        stop_words = set(stopwords.words('english')) 
        
        # Remove retweet text "RT"
        tweet = re.sub(r'^RT[\s]+', '', tweet)
        
        # Remove hyperlinks
        tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
        
        # Remove hash # from words
        tweet = re.sub(r'#', '', tweet)
        
        # Remove cases
        tweet = tweet.lower()

        # Remove punctuation
        tweet = tweet.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize
        tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
        tweet = tokenizer.tokenize(tweet)

        # Remove stopwords
        tweet = [item for item in tweet if item not in stop_words]
        
        # Stem words
        stemmer = PorterStemmer()
        tweet_clean = []
        for token in tweet:
            stem_token = stemmer.stem(token)  # stemming word
            tweet_clean.append(stem_token)
        
        # Merge words
        tweet_clean = ' '.join(tweet_clean)
        
        # Return tweet
        return tweet_clean
