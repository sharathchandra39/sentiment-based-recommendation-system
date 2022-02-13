## Basic REST Enablement For the Sentiment Based Recommendation System. 
# importing libraries 
from logging import log
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import re
import pickle

import sklearn
from nltk.stem import WordNetLemmatizer
lm=WordNetLemmatizer()
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
tf=TfidfVectorizer(max_features=5000,ngram_range=(1,3))
#Import NLP Libraries
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

app = Flask(__name__)
# Intitizalizing and loading the model files which we created as pickle files. 
model = pickle.load(open('model.pkl', 'rb'))
transform_model=pickle.load(open('transform.pkl','rb'))
final_ratings=pickle.load(open('final_ratings.pkl','rb'))
recommend_train=pickle.load(open('traindata.pkl','rb'))   

user_input = None

@app.route("/")
def home():
    #return "Hello from submit page" # render_template('index.html') #Render html page for UI
    return render_template('index.html') #Render html page for UI

@app.route("/submit")
def submit():
    return "Hello from submit page"

# Predict endpoint - takes the user inputs and sends to model and renders the results back.
@app.route('/predict', methods=['POST'])  
def predict():
    user_input = request.form['user_input'] # fetch user input  
    user_input = user_input.lower()     # as we have all lower cases. 
    user_input = user_input  
    print(user_input)
    recommendations = pickle.loads(pickle.dumps(recommend))(user_input)
    print(recommendations)
    if(type(recommendations)==str): 
        recos = []
        for rec in recommendations: 
            recos.append(rec)
        return render_template('index.html', prediction_text='{}'.format(recos))
    else: 
        return (render_template('index.html', user_input = user_input, prediction_text=' 1.{} \n 2.{} 3.{} 4.{} 5.{}'.format(recommendations[0],recommendations[1],recommendations[2],recommendations[3],recommendations[4])))

# Actual Recommendation Definition - an interanl memthod. 
def recommend(user_input):
    try:
        user_recommendations = final_ratings.loc[user_input].sort_values(ascending=False)[0:20] 
        user_recommendations_result = {'product': user_recommendations.index, 'recom_value': user_recommendations.values}
        newdf = pd.DataFrame(user_recommendations_result, index=range(0,20))
        positive_ratings =[]
        for i in range(20):
            positive_ratings.append(sum(recommend_train[recommend_train['name'] == newdf['product'][i]]['user_sentiment'].values)/len(recommend_train[recommend_train['name'] == newdf['product'][i]]))
        newdf['positive_ratings'] = positive_ratings
        newdf.sort_values(['positive_ratings'],ascending=False) 
        # with sort descending - top rating would be first 
        result=newdf.sort_values(['positive_ratings'],ascending=False)[:5]
        result.reset_index(inplace=True)
        recommended = result['product'].values
        return recommended
    except:
        return "No User Available OR No Recommendations for given user."


if __name__ == '__main__':
    app.run()
