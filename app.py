from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import string
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
lemma=WordNetLemmatizer()
#from model import prep
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.externals import joblib
#from model import preprocess
# load the model from disk
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('tranform.pkl','rb'))
app = Flask(__name__)

def prep(text):
    text="".join([i for i in text if i not in string.punctuation])
    text=text.lower()
    tokens = re.split('W+',text)
    output = [lemma.lemmatize(word) for word in tokens if not word in stopwords.words('english')]
    output=' '.join(output)
    return output
@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = pd.Series(message)
        data1=data.apply(lambda x : prep(x))
        vect = cv.transform(data1).toarray()
        my_prediction = clf.predict(vect)
        return render_template('result.html',prediction = my_prediction)
    
if __name__ == '__main__':
	app.run(debug=True)