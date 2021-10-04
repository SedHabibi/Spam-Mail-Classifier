# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 15:34:12 2021

@author: Neelabh
"""
import time
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def mail_classifier(text):
    #Loading the model and vocabulary.
    model = open('model.pkl', 'rb')
    vocabular = open('vocabular.pkl', 'rb')
    forest = pickle.load(model)
    vocab = pickle.load(vocabular)
    
    #Initializing the vectorizer.
    vector1 = TfidfVectorizer(decode_error = 'replace', vocabulary = vocabular)
    new_text = vector1.fit_transform([text])
    
    #Predicting.
    processed_text = forest.predict(new_text)
    return processed_text

if __name__ == '__main__':
    text = input("Enter the SMS : ")
    mail_classifier(text)
    
    if mail_classifier(text)[0] == 0:
        print('Not a spam SMS!')
    else:
        print("Spam mail!")
    
