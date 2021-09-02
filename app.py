from flask import Flask,render_template,request
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

app = Flask("Emailcalssifierapp")

model = pickle.load(open('spam.pkl','rb'))
cv=pickle.load(open('vectorizer.pkl','rb'))

@app.route("/")
def home():
    return render_template("index.html")
@app.route("/result",methods=["GET"])
def result():
    query=request.args.get("email")
    data=[query]
    vec=cv.transform(data).toarray()
    result=model.predict(vec)
    if result[0]==0:
        stm="This is Not A Spam Email"
        htmlcode=render_template("index.html",result=stm)
        return htmlcode
    else:
        stm="This is A Spam Email"
        htmlcode=render_template("index.html",result=stm)
        return htmlcode
app.run(port=1111,host="192.168.99.1")