from flask import Flask,redirect,url_for,render_template,request,jsonify
import pandas as pd
import pickle
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer

app=Flask(__name__)


with open('hate.pickle','rb') as f:
    hates=pickle.load(f)
with open('tfid.pickle','rb') as f:
    tfidf_vectorizer=pickle.load(f)
      


@app.route('/')      
def home():
     return render_template("home.html")

@app.route('/login')
def login():
     return render_template("login.html")

@app.route('/upload')
def upload():
     return render_template("upload.html")


@app.route('/read_data',methods=['post'])
def read_data():
    data=request.files['filedata']
    data=pd.read_csv(data)
    print(data)
    return render_template('preview.html',data=data)



@app.route('/prediction') 
def prediction():
     return render_template('prediction.html')

@app.route('/check')
def check():
     abc=request.args.get('tweet')
     input_data=[abc.rstrip()]
     tfidf_test=tfidf_vectorizer.transform(input_data)
     y_pred=hates.predict(tfidf_test)
     if y_pred[0]==1:
          label='Offensive'
     elif y_pred[0]==0:
          label='Non Offensive'
     return render_template('prediction.html',prediction_text=label)


def load_data():
     df=pd.read_csv('data.csv')
     label_counts=df['label'].value_counts().to_dict()
     return label_counts


@app.route('/data')
def data():
     data=load_data()
     return jsonify(data)

@app.route('/performance')
def performance():
     return render_template('performance.html')  

@app.route('/chart')
def chart():
     return render_template('chart.html')  


if __name__=='__main__':
     app.run(debug=True)

  
