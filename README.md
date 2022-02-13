# Sentimemnt Based Product Recommendation System 

### Author : Sharath Chandra Linga 

First Did with EDA and cleaned up the data and made it ready for the Model Building. 

Then observed that imbalance there - so using SMOTE we corrected the class imbalance. 

Afterthat dataset - went ahead with modelling strategies.

First built the model with various strategies like - Logistic/Rainforest/Naivebayes and found out the Logistic regression one is giving the better results whencompare to others. 

So using TF-IDF oversampled using SMOTE analysis - we have found the below Accuracy Levels for 3 Regression Models.

Logistic Regression: 90%
Random Forest: 89%
Naive Bias: 85%
So here in this one we have Logistic Regression has the highest accuracy with TF-IDF

After finalizing the model - then went ahead with creating the recommendations on two types. 

User-based recommendation system

Item-based recommendation system

Then using the correleation matrix products - and also considering the user sentiment created the recommendations. 

Deployed the App on the Heroku - and the Direct URI is https://sharath-sentiment-based-recomm.herokuapp.com/ 

Possible user names - like mike,joshua 

It gives the Top 5 Recommendations for the valid user - based on the user sentiment. 