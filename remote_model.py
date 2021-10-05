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
    vocab = open('word_of_bag.pkl', 'rb')
    forest = pickle.load(model)
    v = pickle.load(vocab)
    
    #Initializing the vectorizer.
    vector1 = TfidfVectorizer(decode_error = 'replace', vocabulary = v)
    new_text = vector1.fit_transform([text])
    
    #Predicting.
    processed_text = forest.predict(new_text)
    return processed_text

if __name__ == '__main__':
    start_time = time.time()
    text = input("Enter the SMS : ")
    mail_classifier(text)
    
    if mail_classifier(text)[0] == 0:
        print('Not a spam SMS!')
    else:
        print("Spam mail!")
    
    print("--- %s seconds ---" % (time.time() - start_time))                              