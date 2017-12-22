import numpy as np
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from pylab import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
#READING TRAINING DATA 
data = pd.read_csv('(A)TRAIN_SMS.csv', encoding='latin-1')
classification = {"ham":0,"spam":1,"info":2}
data['Label']=data['Label'].map(classification)
#TRAINING DATA 
X_train, X_test, y_train, y_test = train_test_split(data['Message'],data['Label'],test_size=300)
vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(X_train)
classifier = MultinomialNB()
classifier.fit(counts,y_train)

#READING THE TEST DATA
f=pd.read_csv('test1.csv',encoding='latin-1')
#READING THE ACTUAL RESULT
actual result =pd.read_csv('actual.csv',encoding='latin-1')
classification = {"ham":0,"spam":1,"info":2}
actual['Label']=actual['Label'].map(classification)
test=f["Message"]
#PASSING
counts_test = vectorizer.transform(test)
prediction = classifier.predict(counts_test)
#ACCURACY
print('Accuracy score: {}'.format(accuracy_score(actual, prediction)))
d1=pd.DataFrame({"Label":prediction})
classification = {0:"ham",1:"spam",2:"info"}
#OUTPUT WRIITEN TO CSV FILE
d1['Label']=d1['Label'].map(classification)
d1.to_csv("output.csv")
