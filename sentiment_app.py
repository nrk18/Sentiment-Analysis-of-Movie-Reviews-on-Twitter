import pickle
import os
from flask import Flask, jsonify, make_response, request, redirect, render_template

app = Flask(__name__)


vectorizer = pickle.load(open('vectorizer.sav', 'rb'))
classifier = pickle.load(open('classifier.sav', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def sentiment():
    if request.method == 'POST':
        text = request.form.get('text')
        text_vector = vectorizer.transform([text])
        result = classifier.predict(text_vector)
        if result==0:
            return render_template('home.html', message= "Negative")
        else:
            return render_template('home.html', message= "Positive")
    return render_template('home.html')

if __name__ == '__main__':
    app.debug = False
    app.run()