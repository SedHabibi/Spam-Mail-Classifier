# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 08:43:07 2021

@author: Neelabh
"""

#Necessary libraries.
import time 
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

#Model training.
def train_model():
    #Loading data.
    data = pd.read_csv('SMS_cleaned.csv')
    data.dropna(axis = 0, inplace = True)
    #Features and labels.
    features = data.iloc[:,0]
    labels = data.iloc[:,1].values
    x_train, x_test, y_train, y_test = tts(features, labels, train_size = 0.80, random_state = 42)
    
    #Tokenization.
    vector = TfidfVectorizer(min_df = 2)
    vect_x_train = vector.fit_transform(x_train)
    
    #Creating objects for classifiers.
    tree= DecisionTreeClassifier(random_state = 24) 
    forest= RandomForestClassifier(random_state = 24)
    knn= KNeighborsClassifier()
    svm= SVC(random_state = 24)
    xboost= XGBClassifier(random_state=24)
    logistic = LogisticRegression(random_state = 24)
    
    #List for models.
    models = [tree, forest, knn, svm, xboost, logistic]
    
    #Checking for accuracy of each model.
    for model in models:
        model.fit(vect_x_train, y_train)
        prediction = model.predict(vector.transform(x_test))
        score = roc_auc_score(y_test, prediction)
        cm = confusion_matrix(y_test, prediction)
        
        #Printing the output.
        print("\n=======MODEL : " + type(model).__name__ + "=======\n")
        print('\nConfusion Matrix : \n', cm)
        print('\nScore of the model : \n', score * 100)
        print("-" * 50)
        time.sleep(3)

if __name__ == '__main__':
    start_time = time.time()
    train_model()     
    print("--- %s seconds ---" % (time.time() - start_time))                              
        

'''
Model score Info:
=======MODEL : DecisionTreeClassifier=======


Confusion Matrix : 
 [[166  14]
 [  5  31]]

Score of the model : 
 0.8916666666666667
--------------------------------------------------

=======MODEL : RandomForestClassifier=======


Confusion Matrix : 
 [[178   2]
 [  6  30]]

Score of the model : 
 0.9111111111111112
--------------------------------------------------

=======MODEL : KNeighborsClassifier=======


Confusion Matrix : 
 [[180   0]
 [ 26  10]]

Score of the model : 
 0.6388888888888888
--------------------------------------------------

=======MODEL : SVC=======


Confusion Matrix : 
 [[178   2]
 [ 10  26]]

Score of the model : 
 0.8555555555555556
--------------------------------------------------
=======MODEL : XGBClassifier=======


Confusion Matrix : 
 [[168  12]
 [  6  30]]

Score of the model : 
 0.8833333333333334
--------------------------------------------------

=======MODEL : LogisticRegression=======


Confusion Matrix : 
 [[179   1]
 [ 15  21]]

Score of the model : 
 0.788888888888889
--------------------------------------------------
'''
















            