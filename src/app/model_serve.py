import pickle5 as pickle
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

# Instantiate API
app = FastAPI()

class TweetIn(BaseModel):
    tweet: list #str

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

@app.post('/predict')
def predict_multiple(data: TweetIn):    
    dt = pd.DataFrame({'tweet':data.tweet})
    dt['prob'] = model.predict_proba(dt['tweet'])[:,1]
    dt = dt.set_index('tweet')
    return dt.to_dict()
    
## TO run: uvicorn model_serve:app --reload
import uvicorn
if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)