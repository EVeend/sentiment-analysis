import pandas as pd
from transformers import pipeline

class SentimentAnalysis:
    
    def __init__(self):
        self.sentiment_pipeline = pipeline(model="cardiffnlp/twitter-roberta-base-sentiment-latest")

    def predict(self, request):
        return self.sentiment_pipeline(request)
    
    def predict_csv(self, dataframe):
        for index, row in dataframe.iterrows():
            result = self.sentiment_pipeline(row[1])
            dataframe.at[index, 'model_output'] = result[0]['label']
            dataframe.at[index, 'confidence_score'] = result[0]['score']
        print("TYPEE", dataframe.head())
        print("TYPEE", type(dataframe))
        return dataframe