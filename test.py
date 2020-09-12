from keras.models import load_model 
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import re

model = load_model('modelE.pkl')

regex = re.compile("[А-Яа-я:=!\)\()A-z\_\%/|]+")
def words_only(text, regex=regex):
    try:
        return " ".join(regex.findall(text))
    except:
        return ""

with open('tokenizer_yota_modelE.pickle', 'rb') as handle:
    tokenizer_yota = pickle.load(handle)

def show_tonality(text):
    # проверка тональности текста 
    text = words_only(text)
    a = []
    a.append(text)
    pred_Sequences = tokenizer_yota.texts_to_sequences(a) #разбиваем текст на последовательность индексов
    xTrainE_pred = pad_sequences(pred_Sequences, maxlen=400)
    return model.predict_classes((xTrainE_pred))

# print(show_tonality('йота все отлично работает'))