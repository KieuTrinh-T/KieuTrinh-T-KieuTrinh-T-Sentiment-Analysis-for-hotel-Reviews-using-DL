# Sentiment-Analysis-for-hotel-Reviews-using-DL
Input: Reviews crawled from TripAdvisor
Output: Polarity (Positive, Neutral or Negative)
Description: In this project, LSTM and the pretrained model distilBert were applied. For LSTM, we used two data (one is the original dataset with 84.2% positive and the other data was under-resampled) to train the same LSTM architecture in order to measure the bias phenomenon.
Then, a Flask app was run to get input of review and analyse its sentiment.
Result: DistilBert Model gave the highest F1-score at 0.91, the next is LSTM with original data at 0.89 and the last one is slightly lower at 0.88.
## Requirements:
 - pip install pandas
 - pip install tensorflow 
 - pip install keras
 - pip install flask

