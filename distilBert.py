import ktrain
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

predictor = ktrain.load_predictor('Models/distilBert')

def predict(text):
    sent = predictor.predict(text)
    if sent == 'Sentiment_0':
        return 'negative'	
    elif sent == 'Sentiment_1':
        return 'neutral'
    elif sent == 'Sentiment_2':
        return 'positive'