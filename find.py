
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import random as rn
from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
 
df = pd.read_csv(r'C:\Users\Nitin\Desktop\Projects\flipkart.csv')
no_of_products=df.shape[0]
#print (no_of_products)

new_df=df[['pid', 'product_specifications', 'product_name', 'brand', 'retail_price', 'discounted_price', 'overall_rating', 'product_category_tree']]
#print (new_df.shape[1])
#rn.seed(0)
new_df=new_df.replace(to_replace="No rating available", value=rn.uniform(0.1, 5.0))
def find_product_type(cat_tree):
    ans=''
    for i in range(2, len(cat_tree)):
        if cat_tree[i]=='>':
            break
        else:
            ans+=cat_tree[i]
            #print(ans)
    return ans
    
#print (find_product_type(new_df.loc[0, 'product_category_tree']))   
#for i in range(no_of_products):
#    new_df.loc[i, 'product_type']=find_product_type(new_df.loc[i, 'product_category_tree'])

#for i in range(100):
    
#    if(new_df.loc[i, 'overall_rating']=="No rating available"):
#        new_df.loc[i, 'overall_rating']=rn.uniform(0.1, 5.0)
        
#for i in range (50):
#    print(new_df.loc[i].overall_rating)

data_frame=new_df[['brand', 'retail_price', 'discounted_price', 'overall_rating']]
#print (data_frame.shape[1])
X=data_frame.drop('overall_rating', axis=1)
y=data_frame.overall_rating
#print(X_test.shape)
X_train, X_test, y_train, y_test=train_test_split(X, y, random_state=0)

#print (X.head())
#one_hot_data = pd.get_dummies(data_frame[['brand','retail_price','overall_rating']])
one_hot_data = pd.get_dummies(X_train)
one_hot_data = one_hot_data.fillna(method='ffill')
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(one_hot_data, y_train) 
X_test_encoded=pd.get_dummies(X_test)
#print (X_test_encoded.shape)
X_test_encoded=X_test_encoded.fillna(method='ffill')
y_predict=regressor.predict(one_hot_data)
#print(y_train.shape)
#print (y_predict.shape)
#print(metrics.accuracy_score(y_predict, y_train))

 
list_calculate=["Alison", 200, 120]
for_ans = pd.get_dummies(list_calculate)
for_ans = one_hot_data.fillna(method='ffill')
predicted_rating=regressor.predict(for_ans)
print ("The Predicted Rating is :")
print (predicted_rating)
print (predicted_rating[len(predicted_rating)-1])
 

    
    



    
    
        













