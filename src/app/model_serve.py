import pickle5 as pickle
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

# Instantiate API
app = FastAPI()

class TweetIn(BaseModel):
    tweet: str

class TweetPredict(TweetIn):
    pred: dict


# load trained model
with open(r"./app/model.pickle", 'rb') as output_file:
    model = pickle.load(output_file)

@app.get("/")
async def root():
    return {"message": "Twitter Disaster Prediction"}

@app.get("/pipeline")
def return_pipeline():
    return {'pipeline':str(model)}

# Obs: as of now, this method predicts only one string at a time
# TODO: make it possible for multiple tweets
@app.post('/predict')
def predict(dict_tweet: TweetIn):
    tweet_list = list()
    tweet_list.append(dict_tweet.tweet)
    pred = model.predict_proba(pd.Series(tweet_list))[:,1]
    response_object = {'tweet':dict_tweet.tweet, 'probability':str(pred)}
    return response_object

## TO run: uvicorn model_serve:app --reload
import uvicorn
if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)