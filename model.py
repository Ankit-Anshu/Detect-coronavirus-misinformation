import pandas as pd
import numpy as np
import re
import xgboost
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords    
stop_words = set(stopwords.words('english'))
import xgboost as xgb
from xgboost import XGBClassifier
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.pipeline import Pipeline
import pickle

data=pd.read_csv('C:/Users/HP/Desktop/corona_fake.csv')

data["label"]= data["label"].str.replace("Fake", "Fake ", case = False)
data["label"]= data["label"].str.replace("fake", "Fake ", case = False)
data["label"]= data["label"].str.replace("True", "Genuine", case = False)

data.loc[5]['label'] = 'Fake'
data.loc[15]['label'] = 'Genuine'
data.loc[43]['label'] = 'Fake'
data.loc[131]['label'] = 'Genuine'
data.loc[242]['label'] = 'Fake'

data=data.fillna(' ')

data['textdata']=data['text']+' '+data['title']
print(data['textdata'][0])

data['textdata'] = data['textdata'].str.replace('[^\w\s]','')
data['textdata'] = data['textdata'].str.lower()

stop_words = set(stopwords.words('english'))


def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)

def func_preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
    ## clean (convert to lowercase and remove punctuations and characters and then strip)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
            
    ## Tokenize 
    lst_text = text.split()
    ## remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in 
                    lst_stopwords]
                
    ## Stemming 
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]
                
    ## Lemmatisation 
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]
            
    ## back to string from list
    text = " ".join(lst_text)
    return text


data['total'] = data['textdata'].apply(lambda x: func_preprocess_text(x))

data = data.drop("textdata", axis=1)


data.drop("label", axis=1,inplace=True)

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

X_train, X_test, y_train, y_test = train_test_split(data['total'], y, test_size=0.2,random_state=102)
print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)

tfidf_train = tfidf_vectorizer.fit_transform(X_train) 
tfidf_test  = tfidf_vectorizer.transform(X_test)

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.80)  
tfidf_train = tfidf_vectorizer.fit_transform(X_train) 
tfidf_test = tfidf_vectorizer.transform(X_test)

xgb_model = XGBClassifier(random_state=42, seed=2, colsample_bytree=0.6, subsample=0.7)
xgb_model.fit(tfidf_train, y_train)
pred3 = xgb_model.predict(tfidf_test)

print('Accuracy of XGBoost on test set:',accuracy_score(y_test, pred3))


pac = XGBClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.80)),
    ('clf',xgb_model)])

pipeline = Pipeline([
    ('tfid', TfidfVectorizer(stop_words='english', max_df=0.80)),
    ('pac',xgb_model)])

pipeline.fit(X_train, y_train)

with open('tfid.pickle','wb') as f:
    pickle.dump(tfidf_vectorizer,f)
    

with open('model_fakenews.pickle','wb') as f:
    pickle.dump(pac,f)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.80)),
    ('clf',xgb_model)])

with open('model.pickle', 'wb') as handle:
    pickle.dump(pipeline, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(pipeline, open("tfidf.pickle", "wb"))
