from flask import Flask, request, jsonify, render_template
import distilBert as db_model
import lstm as lstm_model

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('index.html')

@app.route('/', methods = ['POST'])
def predict():
    text = request.form['text1'].lower()
    db_sent = db_model.predict(text)
    lstm_sent,un_lstm_sent = lstm_model.lstm_predict(text)

    return render_template('index.html', text1= text,text2=lstm_sent,text3=un_lstm_sent,text4=db_sent)


if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5000, debug = True)