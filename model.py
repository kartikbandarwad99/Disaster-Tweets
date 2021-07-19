import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

df= pd.read_csv("train.csv")
#Features and Labels
lemma=WordNetLemmatizer()
    
def prep(text):
    
    text="".join([i for i in text if i not in string.punctuation])
    text=text.lower()
    tokens = re.split('W+',text)
    output = [lemma.lemmatize(word) for word in tokens if not word in stopwords.words('english')]
    output=' '.join(output)
    return output
    


X = df['text'].apply(lambda x: prep(x))
y = df['target']

# Extract Feature With CountVectorizer
cv = CountVectorizer()
x = cv.fit_transform(X) # Fit the Data
   
pickle.dump(cv, open('tranform.pkl', 'wb'))
    
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
#Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
filename = 'nlp_model.pkl'
pickle.dump(clf, open(filename, 'wb'))
    

