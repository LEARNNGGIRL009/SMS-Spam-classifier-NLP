
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

messages=pd.read_csv('D:/NLP/Spam classifier/pythonProject1/SMSSpamCollection', sep='\t',names=["lable","message"])

ps= PorterStemmer()
corpus=[]
for i in range(len(messages)):
    review=re.sub('^[a-zA-Z]',' ',messages['message'][i])
    review=review.lower()
    review=review.split()
    #print(review)
    review=[ps.stem(word) for word in review if word not in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)
#print(corpus)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000) # random number 2500
X=cv.fit_transform (corpus).toarray()
#print(X)

y=pd.get_dummies(messages['lable'])
#print(y)
y=y.iloc[:,1].values
#print(y)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size= 0.20,random_state= 0)

from sklearn.naive_bayes import MultinomialNB
spam_detect_model= MultinomialNB().fit(X_train,y_train)
y_pred=spam_detect_model.predict(X_test)

from sklearn.metrics import confusion_matrix # check how valid your prediction is compared to test data
confusion_m=confusion_matrix(y_test,y_pred)
print(confusion_m)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score (y_test,y_pred)
print(accuracy)




def print_hi(name):
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
