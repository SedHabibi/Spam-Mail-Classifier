# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 15:13:49 2021

@author: Neelabh
"""
import time
import pickle as p
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split as tts
from sklearn.feature_extraction.text import TfidfVectorizer

def Model_training():
    
    #Loading the data.
    data = pd.read_csv('SMS_cleaned.csv')
    data.dropna(axis = 0, inplace = True)
    
    #Features and labels.
    features = data.iloc[:,0]
    labels = data.iloc[:,1].values
    
    #Splitting the data.
    x_train, x_test, y_train, y_test = tts(features, labels, train_size = 0.80, random_state = 42)
    
    #Tokenizing the data.
    vector = TfidfVectorizer(min_df = 2)
    vect_x_train = vector.fit_transform(x_train)
    
    #Initializing the model.
    forest = RandomForestClassifier(random_state = 24)
    forest.fit(vect_x_train, y_train)
    prediction = forest.predict(vector.transform(x_test))
    score = roc_auc_score(y_test, prediction)
    cm = confusion_matrix(y_test, prediction)
    
    print("\nScore : ",score * 100)
    print("\nConfusion Matrix : \n", cm)
     
    model = open('model.pkl', 'wb')
    p.dump(forest, model)
    
    vocab = open('vocabular.pkl', 'wb')
    p.dump(vector.vocabulary, vocab)
    
    model.close()
    vocab.close()
if __name__ == '__main__':
    start_time = time.time()
    Model_training()
    print("--- %s seconds ---" % (time.time() - start_time))                              
    
    
    
    
    
    
    
    