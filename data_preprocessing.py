# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 07:25:13 2021

@author: Neelabh
"""

#Necessary Libraries.
import re
import time
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#Merging the two csv's.
def merge():
    try:
        df1 = pd.read_csv('SMS_train.csv')
        df2 = pd.read_csv('SMS_test.csv')    
        
        merged_df = df1.append(df2)
        merged_df.to_csv('SMS.csv')
        
        print('\nMerged the CSV files.')
    except Exception as e:
        print(e)


#Cleaning the data.
def data_cleaning():
    try:
        data = pd.read_csv('SMS.csv')
        
        cleaned_mail = []
        for i in range(len(data)):
            #Alphanumeric check.
            mail = re.sub('[^a-zA-Z]', ' ', data.iloc[i,2])
            mail = ''.join(map(str,mail))
            mail = mail.lower()
            mail = mail.split()
            
            #Removing stopwords.
            mail = [word for word in mail if not word in stopwords.words('english')]
            
            #Stemming.
            stemmer = PorterStemmer()
            mail = [stemmer.stem(word) for word in mail]
            
            mail = ' '.join(mail)
            cleaned_mail.append(mail)
        
        #Making new Dataframe for the cleaned data.
        new_data = pd.DataFrame()
        new_data['Mail'] = cleaned_mail    
        
        #Converting our label to bool.
        new_data['Result'] = np.where(data['Label'] == 'Spam',1,0)
        
        #Deleting rows that contain nan.
        new_data.dropna(axis = 0, inplace = True)
        
        #Making a new csv file for the cleaned data.
        new_data.to_csv('SMS_cleaned.csv', index = False)
        print('\nCleaned the data!')
        
    except Exception as e:
        print(e)
    
if __name__ == '__main__':
    try:
        start_time = time.time()
        print('\nMerging the data.')
        merge()
        
        print('\nCleaning the data.')
        data_cleaning()
        
        print("--- %s seconds ---" % (time.time() - start_time))
    
    except Exception as e:
        print(e)