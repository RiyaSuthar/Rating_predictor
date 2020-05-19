from django.shortcuts import render
from django.http import HttpResponse
from .forms import ContactForm
from django.shortcuts import render

import pandas as pd
import numpy as np
import random as rn
# from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
# import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

# Create your views here.
def contact(request):
    if request.method=="POST":
        form=ContactForm(request.POST)
        if form.is_valid():
            input_brand=form.cleaned_data['brand']
            input_product_type=form.cleaned_data['product_type']
            input_retail_price=form.cleaned_data['retail_price']
            input_discounted_price=form.cleaned_data['discounted_price']

            print (input_brand, input_product_type, input_retail_price, input_discounted_price)


            df = pd.read_csv(r'C:\Users\asus\Desktop\Projects\Projects\flipkart.csv')
            no_of_products=df.shape[0]
            #print (no_of_products)

            new_df=df[['pid', 'product_specifications', 'product_name', 'brand', 'retail_price', 'discounted_price', 'overall_rating', 'product_category_tree']]
            #print (new_df.shape[1])

            new_df=new_df.replace(to_replace="No rating available", value=rn.uniform(2.0, 5.0))


            data_frame=new_df[['brand', 'retail_price', 'discounted_price', 'overall_rating']]
            #print (data_frame.shape[1])
            X=data_frame.drop('overall_rating', axis=1)
            y=data_frame.overall_rating
            #print (X.head())

            X_train, X_test, y_train, y_test=train_test_split(X, y, random_state=0)

            one_hot_data = pd.get_dummies(X_train)
            one_hot_data = one_hot_data.fillna(method='ffill')
            regressor = DecisionTreeRegressor(random_state = 0)
            regressor.fit(one_hot_data, y_train)
            # X_test_encoded=pd.get_dummies(X_test)
            # print (X_test_encoded.shape)
            # X_test_encoded=X_test_encoded.fillna(method='ffill')
            # y_predict=regressor.predict(one_hot_data)
            # print(y_train.shape)
            # print (y_predict.shape)
            # print(metrics.accuracy_score(y_predict, y_train))
            list_calculate=[input_brand, input_retail_price, input_discounted_price]
            # print(list_calculate)
            for_ans = pd.get_dummies(list_calculate)
            for_ans = one_hot_data.fillna(method='ffill')
            predicted_rating=regressor.predict(for_ans)
            print((predicted_rating))
            print (predicted_rating[len(predicted_rating)-1])

            #print("HI")

    form=ContactForm()
    return render(request, 'form.html', {'form':form})
