from flask import Flask, request, jsonify, abort, url_for, redirect, render_template, send_file
from keras.models import load_model 
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import pickle
import re
import os
app = Flask(__name__)

regex = re.compile("[А-Яа-я:=!\)\()A-z\_\%/|]+")
def words_only(text, regex=regex):
    try:
        return " ".join(regex.findall(text))
    except:
        return ""

modelE = load_model('modelE.pkl')

with open('tokenizer_yota_modelE.pickle', 'rb') as handle:
    tokenizer_yota = pickle.load(handle)

@app.route('/')
def hello_world():
    return '<h2>Hello, My very Best Friend!!!!!</h2>'

@app.route('/<username>')
def show_user_profile(username):
    # show the user profile for that user 
    return 'User %s' % (username)

@app.route('/tonality_yota/<text>')
def show_tonality(text):
    # проверка тональности текста
    text = words_only(text)
    a = []
    a.append(text)
    pred_Sequences = tokenizer_yota.texts_to_sequences(a) #разбиваем текст на последовательность индексов
    xTrainE_pred = pad_sequences(pred_Sequences, maxlen=400)
    return str(modelE.predict_classes((xTrainE_pred)))

@app.route('/badrequest400')
def bad_request():
    abort(400)

@app.route('/tonality_yota/yota_post', methods=['POST'])
def add_message():

    try:
        # проверка тональности 
        text = request.get_json()
        text = text['data']
        text = words_only(text)
        a = []
        a.append(text)
        pred_Sequences = tokenizer_yota.texts_to_sequences(a) #разбиваем текст на последовательность индексов
        xTrainE_pred = pad_sequences(pred_Sequences, maxlen=400)
        predict = {'Class: ': int(model.predict_classes((xTrainE_pred))[0])}
        return jsonify(predict)

    except:
        return redirect(url_for('bad_request'))

# secret key
app.config.update(dict(
    SECRET_KEY="115215",
    WTF_CSRF_SECRET_KEY="115216"
))

from flask_wtf import FlaskForm
from wtforms import StringField
from wtforms.validators import DataRequired
from flask_wtf.file import FileField, FileRequired
from werkzeug.utils import secure_filename

class MyForm(FlaskForm):
    name = StringField('name', validators=[DataRequired()])
    file = FileField()


@app.route('/submit', methods=('GET', 'POST'))
def submit():
    form = MyForm()
    if form.validate_on_submit():
        # print(form.name.data)

        f = form.file.data
        filename = form.name.data + '.csv'

        # f.save(os.path.join(
        #     filename 
        # ))

        df = pd.read_csv(f, encoding='utf-8')
        df.columns = ['Текст', 'Тональность']
        df_np = df.to_numpy()
        c = []
        churn_values = {0: 'Негатив', 1: 'Нейтрально', 2: 'Позитив'}

        for i in range(df.shape[0]):
            
            text = words_only(df_np[i][0])
            a = []
            a.append(text)
            pred_Sequences = tokenizer_yota.texts_to_sequences(a) #разбиваем текст на последовательность индексов
            xTrainE_pred = pad_sequences(pred_Sequences, maxlen=400)
            c.append(churn_values[modelE.predict_classes((xTrainE_pred))[0]])
            
        df['Тональность 2'] = c

        df.to_csv(filename, index=False)

        return send_file(filename,
                        mimetype='text/csv',
                        attachment_filename=filename,
                        as_attachment=True) 

    return render_template('submit.html', form=form)