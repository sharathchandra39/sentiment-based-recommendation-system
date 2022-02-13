#!/usr/bin/env python
# coding: utf-8

# # <center> Capstone -Sentiment Based Product Recommendation System </center>
# 
# #### By - 
# ### Sharath Chandra Linga (sharathchandra.linga@gmail.com) 

# ### `Problem statement`: 
# 
# The e-commerce business is quite popular today. Here, you do not need to take orders by going to each customer. A company launches its website to sell the items to the end consumer, and customers can order the products that they require from the same website. Famous examples of such e-commerce companies are Amazon, Flipkart, Myntra, Paytm and Snapdeal.
# 
# In order to do this, you planned to build a **sentiment-based product recommendation system**, which includes the following tasks.
# 
# *   Data sourcing and sentiment analysis
# *   Building a recommendation system
# *   Improving the recommendations using the sentiment analysis model
# *   Deploying the end-to-end project with a user interface
# 
# **What needs to be submitted for the evaluation of the project?**
# 
# *   An end-to-end Jupyter Notebook, which consists of the entire code (data cleaning steps, text preprocessing, feature extraction, ML models used to build sentiment analysis models, two recommendation systems and their evaluations, etc.) of the problem statement defined
# *   The following deployment files
#     *   One 'model.py' file, which should contain only one ML model and only one recommendation system that you have obtained from the previous steps along with the entire code to deploy the end-to-end project using Flask and Heroku
#     *   'index.html' file, which includes the HTML code of the user interface
#     *   'app.py' file, which is the Flask file to connect the backend ML model with the frontend HTML code
#     *   Supported pickle files, which have been generated while pickling the models
# 
# 
# 
# 

# # Step 1: `EXPLORATARY DATA ANALYSIS`
# 
# Load the data and process the data for clean and assesble data for model and visualizations and understanding the various features of the Data. 

# ## 1.1 Import Libraries and Dataset
# 
# Import Liraries as and when needed will update here

# In[1]:


## Import Liraries (As and when required)

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

#Visualization
import matplotlib.pyplot as plt
import seaborn as sns

import re
import sklearn

#limiting the dislay
pd.set_option('max_columns',300)
pd.set_option('max_rows',100000)

#NLP Libraries
import nltk
from nltk.corpus import stopwords

from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
ps=PorterStemmer()
lm=WordNetLemmatizer()
nltk.download('wordnet')
nltk.download('stopwords')

tf=TfidfVectorizer(max_features=5000,ngram_range=(1,3))
from sklearn.model_selection import train_test_split


import sklearn.metrics
#import imblearn
from sklearn.metrics import precision_score,recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score,recall_score

from sklearn.naive_bayes import BernoulliNB 

#Word2Vec
import gensim
from gensim.models.word2vec import Word2Vec

# For Deployment ease to use pickle library
import pickle

import imblearn
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE


# In[66]:


# Radom Forest 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score,recall_score


# In[ ]:


## If you are using the data by mounting the google drive, use the following :
from google.colab import drive
drive.mount('/content/gdrive')


# In[2]:


## Local Execution 
#Reading the dataset 
data = pd.read_csv("sample30.csv")

## Google Colab Execution 
#Reading the dataset 
#f = open('/content/gdrive/MyDrive/Colab Notebooks/Capstone/sample30.csv')  
#data = pd.read_csv(f)
data.shape # Verify count of data.


# ## 1.2 Data Cleaning and Processing
# 
# Remove unwanted data and adjust the missing data or correct with impute averages etc and some visualizations of data available 

# In[3]:


# Print Data to see how data looks like 
data.head(5)


# In[4]:


# Checkign the Null Values in percentage terms. 
((data.isnull().sum()/data.shape[0])*100).round(2)


# It has bee observed that out of `15` columns - `4` columns has more null or empty data. Lets treat them one by one and also Reviews_userCity and review_userProvince has almost all empty data. and also reviews_didPurchase has more than 45% of data null or empty. By seeing the columns and it stats with the values - `Lets drop these columns` for the better evaluations. 

# In[5]:


# Dropping the columns reviews_didPurchase reviews_userCity reviews_userProvince        
data.drop(['reviews_didPurchase','reviews_userCity','reviews_userProvince'],axis=1,inplace=True)


# In[6]:


data.shape # should have 12 columns as we dropped 3. 


# In[7]:


# Checkign the Null Values in percentage terms. 
((data.isnull().sum()/data.shape[0])*100).round(2)


# In[8]:


# Now lets treat - reviews_doRecommend     
data.reviews_doRecommend.value_counts()


# In[9]:


#Based on the values for this reviews_doRecommend filed it has been observed that majority 92% or ~26k of True. Lets impute the missing with True. 
data['reviews_doRecommend'].replace(np.NaN,data['reviews_doRecommend'].value_counts().index[0],inplace=True)


# In[10]:


# Checkign the Null Values in percentage terms. 
((data.isnull().sum()/data.shape[0])*100).round(2)


# In[11]:


#Lets treat now manufacturer           
data.manufacturer.value_counts().head(5)
#It observed that it has very distinct and large no of difrent manufacturers are present. 


# In[12]:


#lets impute the highest manufacturer value for all the null values as the count are very negligible. 
data.manufacturer.replace(np.NaN,data.manufacturer.value_counts().index[0],inplace=True)


# In[13]:


# Checkign the Null Values in percentage terms. 
((data.isnull().sum()/data.shape[0])*100).round(2)


# In[14]:


#Lets repeat the same for review_data, reviews_title,review_username with the highest of the respective columns'data as the missing values are vry less < 1% 
data.reviews_date.replace(np.NaN,data.reviews_date.value_counts().index[0],inplace=True)
data.reviews_title.replace(np.NaN,data.reviews_title.value_counts().index[0],inplace=True)
data.reviews_username.replace(np.NaN,data.reviews_username.value_counts().index[0],inplace=True)


# In[15]:


# Checkign the Null Values in percentage terms. lets check without any roundoffs. 
((data.isnull().sum()/data.shape[0])*100)


# In[16]:


## Still user_sentiment has a very less null values - lets see what are they 

data.info()


# In[17]:


## Only 1 record has empty value. Good to delete that row as the impact would be very less. 

data.dropna(how='any',axis=0,inplace=True)
data.reset_index(drop=True,inplace=True) #has it deletes the entry - we have to reset the index. 


# In[18]:


# Checkign the Null Values in percentage terms. lets check without any roundoffs. 
((data.isnull().sum()/data.shape[0])*100)


# Now the data has no missing values for all the `12` Columns. Lets check the datatypes too. 

# In[19]:


data.info()


# All the columns has the properly assigned with correct datatypes. So no change required. 

# ## 1.3 Feature Validations 
# Understand each column and come up with visulaization how they represent the data and how it can be used for the Sentiment Analysis. 
# 
# 

# In[20]:


data.info() 


# In[21]:


# Common Utility Method for Bar/plots for Visualization 

def show_in_bar_plot(colName):
    plt.figure(figsize=(8,8))
    data[colName].value_counts(normalize=True).plot(kind="bar")
    plt.title("Category's Percentage in the {} Feature")
    plt.xlabel(colName)
    plt.ylabel('Percentage (%)')
    plt.show()


# In[22]:


# We have total 12 Columns - and lets analyze some important things  

# Lets start with ID and Name as both sounds similar and lets check whether they both have the values unique or not.

print('ID Counts ', data.id.nunique())
print('Name Counts ', data.name.nunique())


# In[23]:


# observed that they both have the same unique values so we can assume and 
#now lets see the `Rating categories` i.e. Reviews_rating so that we can see how many are there and how they have values in each category 
(data.reviews_rating.value_counts(normalize=True)*100).round(2)


# In[24]:


show_in_bar_plot('reviews_rating')


# Based on above visualizations - it is observed that Rating 5 is being given for most of the observations. Its upto 70% 

# now lets see how many total brands we have 

# In[25]:


len(data.brand.unique())


# we have over 214 unique brands in total 
# lets visualize the top brands. 

# In[26]:


#since we have to send only top 20 we will use inline visualization for graph / bar
plt.figure(figsize=(14,8))
plt.title('Top 20 Brands')
data.brand.value_counts()[:20].plot(kind='bar')
plt.xlabel('Brand')
plt.ylabel('Count of Reviews')
plt.show()


# In[27]:


#Lets see the same for Categories. 
len(data.categories.unique())


# In[28]:


data.categories.value_counts().head(5)


# It has been observed that these top 5 categories has large set of names seperated by comma - so lets go with the same and later will see if this can be used  in sentiment or not. 

# In[29]:


#Now lets amalyze manufacturers - 

data.manufacturer.value_counts()


# Top 5 manufacturers are 
# *   Clorox
# *   Test
# *   AmazonUs/CLOO7
# *   L'oreal Paris
# *   Walt Disney

# In[30]:


# now lets check for dates - before checking the dates - its good to have the Years which can extract from the date and yarly would be great to see the results. 

# Creating a new column year for ease usage. 
data['year']=  [i.split('-') [0] for i in data.reviews_date]


# In[31]:


data['year'].value_counts().index # to show only the year values 


# In[32]:


# last 6 years of data
plt.figure(figsize=(20,10))
sns.barplot (data[ 'year'].value_counts ().index[:6], data[ 'year'].value_counts().values [:6])
plt.show()


# Year 2014 the highest given rating year and later years it got decreased. 
# 

# In[33]:


show_in_bar_plot('user_sentiment')


# Overall we have above 80% of positive rating. 

# In[34]:


#Lets check how many total users are there unique 

len(data['reviews_username'].value_counts())


# we do have a lot of unique users given the rating which is good sing for the sentiment analysis as we can consider broader audience perspective. 

# Based on above analysis - found out that - we do have 3 important columns i.e. `reviews_title`, `reviews_text` and `usersentiment` and also the remaining columns we use as when needed. Lets proceed for some Text processing. 

# # Step 2:  `TEXT PRE-PROCESSING`
# 
# We use text preprocessing and prepare for model ready data 

# ## 2.1 Text processing using NLTK approaches. 
# Fix review column text data 

# So observed that we do have reviews title and reviews text which has splitted into two columns - lets concatenate it and add it to new column review and drop the rest 2 

# In[35]:


data['review'] = data['reviews_title']+'_'+data['reviews_text']  #Combined with _ now.
data.drop(['reviews_title','reviews_text'], axis=1,inplace=True)  #dropping the columns as we combined/concatenated. 


# In[36]:


data.info() # lets see what all we have now 


# In[37]:


# user_sentiment is having positive or negative - lets go with 0 negative n 1 for positive 
data.user_sentiment =data.user_sentiment.replace({'Positive':1,'Negative':0})


# In[38]:


data.head(3)


# In[39]:


#Lets create a copy of Data so far has - or Clone a copy of data and keep it in - data2 
data2 = data[:]  # : mean all data 


# So lets process the actual text and be prepare the proper data for sentiment analysis for that we have to follow the below various steps: 
# 
# 
# 1.   Stopword Removal and Puntuation Removal
# 2.   Stemming and Lemmatization
# 3.   convert the text to tokens using one of the approaches we have (bag of words, tfidf etc, word2vec etc)
#  
# 
# 

# In[40]:


# Lets start with Stopword removal and correcting the punctuation - will go by english language

processed_reviews = []

for review_sentence in data.review:
  # lets lower all the review string 
  review_sentence = review_sentence.lower()

  # now removing the extra regex from the sentence - we already concatenated with _ but thats not required. 
  review_sentence = re.sub('[^a-zA-Z0-9]',' ',review_sentence)

  # join the sentences 
  review_sentence=review_sentence.split()
  review_sentence=' '.join(review_sentence)  # again for splitting with standard "  "
  review_sentence=review_sentence.split()

  # Now stop word along with Lemmatization 
  review_sentence=[lm.lemmatize(word) for word in review_sentence if word not in set(stopwords.words('english'))]

  #add to the list of processed reviews 
  processed_reviews.append(' '.join(review_sentence))



# In[41]:


processed_reviews


# In[42]:


data['processed_reviews'] = processed_reviews
data.head()


# So far we have done some basic processing of the text using nlp techniques to have processed reviews. Now lets see how many words in each review there. and so that we can see. Lets use quantile approcah to see how many words are there in each review. 
# 
# 

# In[43]:


#using numpy quantile - lets see 
processed_reviews_word_count = []
for i in processed_reviews: 
    processed_reviews_word_count.append((len(i.split())))  # adding the length of the words to a new list 
np.quantile(processed_reviews_word_count,[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.98,0.99,0.995,0.998,0.999,1])  # 0 to 1 with various quantiles. 


# Using above quantiles - observed that - at 98% quantile we have around 140 words and 173 in 99th quantile - so lets go with a standard valu ein between these 150 and will remove the reviews which are more than 150 words. 

# In[44]:


totalReviews = len(data.processed_reviews)
review_index_with_150_words = []

for i,j in zip(range(totalReviews + 1) ,  data.processed_reviews):
    if(len(j.split())>150):  
        if(j == (data.processed_reviews[i])):
            review_index_with_150_words.append(i)    

print('Total Reviews Count with more than 150 words', len(review_index_with_150_words))


# In[45]:


print(' what is the % of more than 150 words out of dataset ')
(len(review_index_with_150_words)/data.shape[0])*100


# It is concluded that we have way negligible number of reviews which is having more than 150 words in it i.e. 0.16% - so lets drop those rows. 
# 

# In[46]:


data.drop(review_index_with_150_words,inplace=True)
data.reset_index(inplace=True) #reset as after delete indexs will be not in correct order 


# In[47]:


data.shape


# So far we have deleted 1 + 48 Records from the given Dataset as per the analysis and understandings how data is there in that 

# In[48]:


# clone the so far data 
data2 = data[:]


# ## 2.2 Correcting the Oversampling
# 
# In order to extract features from the text data, you may choose from any of the methods, including bag-of-words, TF-IDF vectorization or word embedding.

# In[49]:


# lets create a dependent variable i.e. reviews which is processed and user sentimetn as x and y to create the model 

x= data.processed_reviews
y= data.user_sentiment 


# In[50]:


# lets see how data looks like 
print(x.head(3)) 
print(y.head(3))


# Now we have to use either TFIDF / Bagofwords / CountVectorization 
# 
# Ref: https://www.analyticsvidhya.com/blog/2021/07/bag-of-words-vs-tfidf-vectorization-a-hands-on-tutorial/ 
# 
# First Lets go with TF-IDF and see 

# In[51]:


# TF-IDF: Term frequency - Inverse document frequency 
# TFIDF works by proportionally increasing the number of times a word appears in the document but is counterbalanced by the number of documents in which it is present

x=tf.fit_transform(x).toarray()
x


# In[52]:


unique_random_state_for_book = 39


# In[53]:


# create variables with with 70+30 size train and test splits. 

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=unique_random_state_for_book,stratify=y) 


# In[54]:


print('x Train', x_train.shape, 'x Test', x_test.shape)
print('Y Train', x_train.shape, 'Y Test', x_test.shape)


# In[55]:


y_train.value_counts(normalize=True)
y_test.value_counts(normalize=True)


# So the target variable has the very highly imbalanced where 89 and 11 % in train n test  so lets treat this Class Imbalance 

# `**Oversampling**` is capable of improving resolution and signal-to-noise ratio, and can be helpful in avoiding aliasing and phase distortion by relaxing anti-aliasing filter performance requirements
# 
# Some techniques to handle the oversampling 
# 1.   Random Sampling
# 2.   SMOTE 
# 3.   ADASYN
# 
# most popular and perhaps most successful oversampling method is SMOTE; that is an acronym for Synthetic Minority Oversampling Technique
# 
# Ref: https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/ 
# 
# 

# In[56]:


# API Ref: https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html 
smt = SMOTE(0.5,random_state=unique_random_state_for_book)  
# 0.5 for the majority of the data has the values same as minority 
x_train_SMOTE, y_train_SMOTE = smt.fit_resample(x_train, y_train)


# In[57]:


print(x_train_SMOTE.shape, '---', y_train_SMOTE.shape)
print('Actual Y Train stats')
print(y_train.value_counts())
print('After Oversampling the Y Train Counts')
print(y_train_SMOTE.value_counts())


# Now we have the good samples with corrected class imbalance values. 

# # Step 3: `TRAINING A CLASSIFICATION MODEL`
# You need to build at least three ML models. You then need to analyse the performance of each of these models and choose the best model. At least three out of the following four models need to be built (Do not forget, if required, handle the class imbalance and perform hyperparameter tuning.). 
# 1. Logistic regression
# 2. Random forest
# 3. XGBoost
# 4. Naive Bayes

# First Lets go with TF-IDF processed and 3 models from the above 4 regression model. 
# 

# ## 3.1 Logistic Regression
# Logistic regression is a process of modeling the probability of a discrete outcome given an input variable. ... Logistic regression is a useful analysis method for classification problems, where you are trying to determine if a new sample fits best into a category

# In[58]:


lr=LogisticRegression()


# In[59]:


# call fit fuction with oversampled dataset 
lr.fit(x_train_SMOTE,y_train_SMOTE)


# In[60]:


y_pred=lr.predict(x_test)
y_pred


# In[61]:


print('Accuracy for Logistic Regression', accuracy_score(y_test, y_pred))


# `90%` for Logistic Regression 

# In[62]:


print(confusion_matrix(y_test,y_pred))


# In[63]:


print(classification_report(y_test, y_pred))
print("precision score",precision_score(y_test,y_pred))
print("Recall score",recall_score(y_test,y_pred))


# Conclusion: With abvoe various metrics - it is obseved that we have `90%` accuracy with Logistic Regression. Lets check the other model Random Forest 

# ## 3.2 Random Forest 
# Random Forest Regression is a supervised learning algorithm that uses ensemble learning method for regression. 
# A random forest is a meta estimator that fits a number of classifying decision trees on various sub-samples of the dataset

# In[67]:


# Ref: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# Ref: https://machinelearningmastery.com/random-forest-ensemble-in-python/ example 
rc=RandomForestClassifier(random_state=unique_random_state_for_book,n_jobs=-1) #n_jobsint The number of jobs to run in parallel.  -1 means using all processors.


# In[68]:


# since we using gridsearchcv - lets explore https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html 
# and https://www.analyticsvidhya.com/blog/2021/06/tune-hyperparameters-with-gridsearchcv/ 

params = {'max_depth': [1, 2, 5, 10, 20], 'min_samples_leaf': [5, 10, 20, 50, 100],
    'max_features': [2,3,4], 'n_estimators': [10, 30, 50, 100, 200,400]
}


# In[69]:


grid_search_model = GridSearchCV(estimator=rc, param_grid=params,cv=4, n_jobs=-1, verbose=1, scoring = "accuracy")
# 4 fold cross validation  (marking it as 1 for local testing )
#1. estimator – A scikit-learn model
#2. param_grid – A dictionary with parameter names as keys and lists of parameter values.
#3. scoring – The performance measure. For example, ‘r2’ for regression models, ‘precision’ for classification models.
#4. njobs – all processes at parallel running 


# In[117]:


# to print time 

grid_search_model.fit(x_train_SMOTE,y_train_SMOTE)


# In[118]:


rf_best = grid_search_model.best_estimator_
rf_best


# In[119]:


rc=RandomForestClassifier(max_depth=20, max_features=4, min_samples_leaf=10,n_estimators=10, n_jobs=-1, random_state=unique_random_state_for_book)
rc


# In[120]:


rc.fit(x_train_SMOTE,y_train_SMOTE)


# In[121]:


y_pred=rc.predict(x_test)
y_pred


# In[122]:


print("Accuracy: ",accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test, y_pred))
print("precision score: ",precision_score(y_test,y_pred))
print("Recall score: ",recall_score(y_test,y_pred))


# Conclusion: With abvoe various metrics - it is obseved that we have `89`% accuracy with Random Forest Regression. Lets check the other model Navie Bias - XGBoost has issues with deployment as per discussion over class.

# ## 3.3 Naive Bayes
# 
# The Naive Bayes is a classification algorithm that is suitable for binary and multiclass classification. Naïve Bayes performs well in cases of categorical input variables compared to numerical **variables**

# In[72]:


#Like MultinomialNB, this classifier is suitable for discrete data.
nbc=BernoulliNB()


# In[73]:


nbc.fit(x_train_SMOTE,y_train_SMOTE)


# In[74]:


y_pred=nbc.predict(x_test)
y_pred


# In[75]:


print("Accuracy: ", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test, y_pred))
print("precision score",precision_score(y_test,y_pred))
print("Recall score",recall_score(y_test,y_pred))


# Conclusion: With abvoe various metrics - it is obseved that we have `85`% accuracy with Naive Bias Regression. 

# So using TF-IDF oversampled using SMOTE analysis  - we have found the below Accuracy Levels for 3 Regression Models. 
# 
# 1.   Logistic Regression: `90%`
# 2.   Random Forest: `89%`
# 3.   Naive Bias: `85%`
# 
# So here in this one we have Logistic Regression has the highest accuracy with TF-IDF. 
# 
# ### Lets go with TF-IDF Model with Logistic Regression which has the better metrics for the model. 

# # Step 4: `BUILDING A RECOMMENDATION SYSTEM`
# 
# you can use the following types of recommendation systems.
# 
#  
# 
# 1. User-based recommendation system
# 
# 2. Item-based recommendation system

# ##4.1.1 User Based Recommendation System 
# 

# In[76]:


# Create a clone copy 
data_ubr = data[:] # User Based Recommendation  UBR
data_ubr.info()


# In[77]:


# For Recommendations Split data into train and test 
recommend_train, recommend_test = train_test_split(data_ubr, test_size=0.3, random_state=unique_random_state_for_book)


# In[78]:


print("Recommndation training data:",recommend_train.shape)
print("Recommndation testing data: ",recommend_test.shape)


# In[79]:


pickle.dump(recommend_train,open('traindata.pkl','wb'))


# As we have seen there are almost 14 columns or data features there and we have created the model with review text and sentiment - and now for user based recommendation lets go with relative columns ie. `user name, rating and product name`

# In[80]:


#creating pivot - with above columns. 
df_pivot = recommend_train.pivot_table(index='reviews_username',values='reviews_rating',columns='name').fillna(0)
df_pivot.head(5)


# Lets gathers the products which were not rated by the user. 
# 
# 

# In[81]:


def not_rated_products_user(x):
  if x>=1:
      return 0
  else:
      return 1


# In[82]:


recommendtrain_2 = recommend_train.copy()
recommendtrain_2['reviews_rating'] = recommendtrain_2['reviews_rating'].apply(not_rated_products_user)


# In[83]:


recommendtrain_2 = recommendtrain_2.pivot_table(index='reviews_username',values='reviews_rating',columns='name').fillna(1)


# In[84]:


recommendtrain_2.head(5)


# Cosine similarity measures the similarity between two vectors of an inner product space
# 
# Adjusted cosine similarity measure is a modified form of vector-based similarity where we take into the fact that different users have different ratings schemes

# In[85]:


#adjusted cosine similarity 
from sklearn.metrics.pairwise import pairwise_distances
usercorrelation = 1 - pairwise_distances(df_pivot, metric='cosine')
usercorrelation[np.isnan(usercorrelation)] = 0
print(usercorrelation)


# In[86]:


#Similarity matrix 
usercorrelation[usercorrelation<0]=0
usercorrelation


# In[87]:


user_predicatd_ratings = np.dot(usercorrelation, df_pivot.fillna(0))
user_predicatd_ratings


# Since we are looking for only products not rated by user lets ignore the rest 

# In[88]:


final_ratings_user = np.multiply(user_predicatd_ratings,recommendtrain_2)
final_ratings_user.head()


# ## 4.1.2 Finding the Top5 User Recos 

# In[89]:


user_name_input = input("Please enter Username: ")
print(user_name_input)


# In[90]:


pickle.dump(final_ratings_user,open('final_ratings.pkl','wb'))


# ## 4.1.3 Recommendations for the Given Username

# In[91]:


top_20_recommendations_user = final_ratings_user.loc[user_name_input].sort_values(ascending=False)[0:20]
print('Top 20 Recommendations for User ', user_name_input)
top_20_recommendations_user


# `**`Conclusion - thats the user to user recommendation where we followed cosine similarity and dot products to fetch the recommendations. **

# ## 4.2.1 Item Based User Recommendation System

# In[92]:


# Create a clone copy 
data_ibr = data[:] # User Based Recommendation  UBR
data_ibr.info()


# In[93]:


df_itembased_pivot = recommend_train.pivot_table(index='reviews_username',values='reviews_rating',columns='name').T
df_itembased_pivot.head()


# In[94]:


# Adjusting the cosine for the nromalization 
normalizedMean  = np.nanmean(df_itembased_pivot, axis=1) # on column1 
final_df_itembased_pivot = (df_itembased_pivot.T-normalizedMean).T


# In[95]:


final_df_itembased_pivot.head(3)


# Now we jave adjusted the cosine on the items that has been reviewed - lets check the Item Simialrity Matrix 

# In[96]:


item_corr_itembased = 1 - pairwise_distances(final_df_itembased_pivot.fillna(0), metric='cosine')
item_corr_itembased[np.isnan(item_corr_itembased)] = 0
print(item_corr_itembased)


# we need only values where greater than 0 so that we can get the positive ones. 

# In[97]:


item_corr_itembased[item_corr_itembased<0]=0
item_corr_itembased


# ## 4.2.2 Item Prediction

# In[98]:


#check  the shapes for correct dot product 
print(final_df_itembased_pivot.shape)
print(item_corr_itembased.shape)


# In[99]:


#Now we have to do the dot product with the above correlation to get the matrix 

item_prediction_ratings = np.dot((final_df_itembased_pivot.fillna(0).T),item_corr_itembased)
item_prediction_ratings


# In[100]:


#check  the shapes for correct dot product 
print(item_prediction_ratings.shape)
print(recommendtrain_2.shape)


# In[101]:


#Now final rating for items with products not rated by users. 
final_item_item_ratings = np.multiply(item_prediction_ratings,recommendtrain_2)
final_item_item_ratings.head(3)


# ## 4.2.3 Finding the `Top 5` Recos for Given User

# In[102]:


user_name_input = input("Please enter Username: ")
print(user_name_input)


# In[103]:


top_20_recommendations_item = final_item_item_ratings.loc[user_name_input].sort_values(ascending=False)[0:20]
print('Top 20 Recommendations for User ', user_name_input)
top_20_recommendations_item


# `**`Conclusion - thats the item to item recommendation where we followed cosine similarity and dot products to fetch the recommendations. **

# # Step5 : `FINE TUNING THE RECOMMEDNATIONS BASED ON THE USER SENTIMENT`
# 
# Predicted the sentiment (positive or negative) of all the reviews in the train data set of the top 20 recommended products for a user. For each of the 20 products recommended, found the percentage of positive sentiments for all the reviews of each product. Filtered out the top 5 products with the highest percentage of positive reviews

# In[104]:


# top 20 user - user recommendations: for Mike 
top_20_recommendations_user


# In[105]:


# top 20 item - item recommendations: For Mike username 
top_20_recommendations_item


# Now lets filiter among these based on the `positive` Reviews - represent top 5 recos . 

# In[106]:


# lets create the dataframes with above results for both u-u and i-i based results. 
user_recos = {'Product_Name': top_20_recommendations_user.index, 'Recommendation': top_20_recommendations_user.values}
item_recos = {'Product_Name': top_20_recommendations_item.index, 'Recommendation': top_20_recommendations_item.values}


# In[107]:


# converting the above list to dataframes 
user_recos_df=pd.DataFrame(user_recos,index=range(0,20))
user_recos_df


# In[108]:


# converting the above list to dataframes 
item_recos_df=pd.DataFrame(item_recos,index=range(0,20))
item_recos_df


# In[109]:


# First User Recos: alogng with Sentimetn 
user_positive_ratings = [] 
for i in range(20):
    user_positive_ratings.append(
        sum(recommend_train[recommend_train['name'] == user_recos_df['Product_Name'][i]]['user_sentiment'].values)/len(recommend_train[recommend_train['name'] == user_recos_df['Product_Name'][i]])
    )


# In[110]:


# Above sum values will be added as new column - 
user_recos_df['Positivity'] = user_positive_ratings


# In[111]:


user_recos_df.head(20) # printing all values 


# In[112]:


# Now lets sort with Descending order to print top 5 based on user sentiment too. 
user_recos_df.sort_values(['Positivity'],ascending=False).head(5)


# Now lets repeat a few above steps for Item-Item based also based on positivity 

# In[113]:


# First Item Recos: alogng with Sentimetn 
item_positive_ratings = [] 
for i in range(20):
    item_positive_ratings.append(
        sum(recommend_train[recommend_train['name'] == item_recos_df['Product_Name'][i]]['user_sentiment'].values)/len(recommend_train[recommend_train['name'] == item_recos_df['Product_Name'][i]])
    )


# In[114]:


# Above sum values will be added as new column - 
item_recos_df['Positivity'] = item_positive_ratings
item_recos_df.head(20)


# In[115]:


# Now lets sort with Descending order to print top 5 based on item based user sentiment too. 
item_recos_df.sort_values(['Positivity'],ascending=False).head(5)


# In[123]:


#Creating the pickle dumps final 
pickle.dump(lr,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
pickle.dump(tf,open('transform.pkl','wb'))


# In[ ]:




