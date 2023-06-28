import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

import re
import nltk
import nltk
nltk.download('stopwords')
import nltk
nltk.download('snowball_data')

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from wordcloud import WordCloud


#For intrest sake \t stands for tab \n->new line 
df= pd.read_csv("SMSSpamCollection",sep="\t", names=['label', 'text'])
df.head()

#reading the first 10 texts
for i in range(10):
    print(df.iloc[i,-1])
    print()

####Cleaning data###

#Stopwords are frequently occuring words that carry insignificant meaning
#Removing stop words help reduce the noise and improve the efficiency of subsequent tasks
sn=SnowballStemmer("english")
stop = set(stopwords.words('english')) #use set so you can only get unique ones
sms='I love Machine Leaarning <><<><KPK 0283020 lsndhn' #it's just an example but in a nutshell you will be cleaning most of the text

def clean_text(sms):
    sms= sms.lower()  #lower FREE and free have different demensions on the vector space. plus they will be treated differently
    #sms=re.sub(("[^a-z0-9]", '', sms)) #make sure you remove all the unnecessary characters
    sms=nltk.word_tokenize(sms)
    sms=[t for t in sms if len(t)>1] #removing all values wit length zero like zero
    sms=[sn.stem(word) for word in sms if word not in stop] # if the word is not in stop word then save it 
    #if the sms contains for example playing play then steam=>sn.stem(word) it into one character-> play
    sms=' '.join(sms)
    return sms

df['clean_text']= df['text'].apply(clean_text) #add an extra col of clean text

hamdata= df[df['label']=='ham']

hamdata=hamdata['clean_text']

#WordCloud- A graphical representation of a lot of words- Graphical Interpretation of a lot of words
#The words that appear a lot will be larger
for i in range(10):
    print(df.iloc[i,-1])
    print()

def worldCloud(data):
    words=''.join(data)
    wc=WordCloud(background_color='white')
    wc=wc.generate(words)
    plt.figure(figsize=(10,8))
    plt.imshow(wc)
    plt.show
worldCloud(hamdata)

spamdata=df[df['label']=='ham']
spamdata=spamdata['clean_text']

#Converting text data into numeric numbers
#text-featurization
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000)
X=cv.fit_transform(df['clean_text']).toarray()
y=pd.get_dummies(df['label'])
y=y['spam'].values
print("-",pd.get_dummies(pd.get_dummies(df['label'])))
print(X.shape)


print('x', X)
print('y', y)

#Model Bulding
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
actuals=model.fit(X_train, y_train)
y_pred=model.predict(X_test)

print("actuals", y_test)
print("predicted", y_pred)

print(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).head(10)) #can remove head(10) to see all of them

#Model evaluation

print("Accuracy on training data: ")
print(model.score(X_train, y_train))


print("Accuracy on test to data: ")
print(model.score(X_test, y_test))


from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, y_pred))

#[[949   6] -> out of 956 test data are 949 are predicted as non spam -> made 7 mistakes 
# [  7 153]] -> the spam data out of 159 -> 153 is predicted correctly only 6 of them are predicted wrong
# so in total 13 values are predicted wrong

print(len(y_test))
print(classification_report(y_test,y_pred))

#print(spamdata)





