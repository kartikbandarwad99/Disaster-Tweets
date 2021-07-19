from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from model import prep
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.externals import joblib
#from model import preprocess
# load the model from disk
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('tranform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        message=message.lower()
        data = pd.Series(message)
        data1=data.apply(lambda x : prep(x))
        vect = cv.transform(data1).toarray()
        my_prediction = clf.predict(vect)
        return render_template('result.html',prediction = my_prediction)
    
if __name__ == '__main__':
	app.run(debug=True)