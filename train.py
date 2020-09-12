import re
import time
import keras
import pandas as pd
import numpy as np
from time import time
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from keras.models import load_model 

from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

from keras.layers import Embedding, Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Masking, Convolution1D, concatenate, GlobalMaxPooling1D, BatchNormalization, SpatialDropout1D, SpatialDropout1D, SimpleRNN, GRU, LSTM
from keras.models import Model, Sequential
from keras_tqdm import TQDMNotebookCallback

import random
random.seed(1228)


# функция для удаления лишних символов в текстах
regex = re.compile("[А-Яа-я:=!\)\()A-z\_\%/|]+")

def words_only(text, regex=regex):
    try:
        return " ".join(regex.findall(text))
    except:
        return ""

df = pd.read_excel('Тональность_метки.xlsx')
df_result_df = df[['Дайджест текста', 'Тональность']]
df_result_df.columns = ['Текст', 'Тональность']
print(df_result_df.shape)

# удаляем лишние символы, оставляем слова
df_result_df['Текст'] = df_result_df['Текст'].apply(words_only)

X = df_result_df['Текст'].tolist()
y = df_result_df['Тональность'].tolist()

X, y = np.array(X), np.array(y)

X_text_train, X_text_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
print("Данных на обучение: ", len(y_train))
print("Данных на тест: ", len(y_test))

# Посчитаем максимальную длину текста описания в словах
max_words = 0
for desc in df_result_df['Текст'].values:
    words = len(desc)
    if words > max_words:
        max_words = words
print('Максимальная длина сообщения: {} слов'.format(max_words))

# Подготовка данных
MaxWordCount = 5000 # сколько слов возьмем  
tokenizer = Tokenizer(num_words=MaxWordCount, filters='!"#$%&()*+,-–—./:;<=>?@[\\]^_`{|}~\t\n\xa0...', lower=True, split=' ', oov_token='unknown', char_level=False)
tokenizer.fit_on_texts(X_text_train)

sequences_train = tokenizer.texts_to_sequences(X_text_train) #обучающие тесты в индексы
X_train = pad_sequences(sequences_train, maxlen=500)
sequences_test = tokenizer.texts_to_sequences(X_text_test) #проверочные тесты в индексы
X_test = pad_sequences(sequences_test, maxlen=500)

# баланс классов 
print('Распределение классов: ', df['Тональность'].value_counts())

xTrain01 = tokenizer.sequences_to_matrix(X_train.tolist())#подаем xTrain в виде списка чтобы метод успешно сработал
xTest01 = tokenizer.sequences_to_matrix(X_test.tolist())#подаем xTest в виде списка чтобы метод успешно сработал
print(xTrain01.shape)       #Размер обучающей выборки, сформированной по Bag of Words
print(xTrain01[0][0:100]) #фрагмент набора слов в виде Bag of Words

# кодирование признаков
le = LabelEncoder()
le.fit(['Нейтральная', 'Негативная', 'Позитивная'])
y_train_cat = np_utils.to_categorical(le.transform(y_train), 3)
y_test_cat = np_utils.to_categorical(le.transform(y_test), 3)

word_index = tokenizer.word_index
print('Всего слов в словаре: ', len(word_index))

# Нейросеть
#Создаём полносвязную сеть
model01 = Sequential()
model01.add(BatchNormalization())
model01.add(Dense(200, input_dim=MaxWordCount, activation="relu"))
model01.add(Dropout(0.25))
model01.add(BatchNormalization())
model01.add(Dense(3, activation='sigmoid'))

model01.compile(optimizer='adam', 
            loss='categorical_crossentropy', 
            metrics=['accuracy'])

#Обучаем сеть на выборке, сформированной по bag of words - xTrain01
history = model01.fit(xTrain01, 
                    y_train_cat, 
                    epochs=20,
                    batch_size=64,
                    validation_data=(xTest01, y_test_cat), 
                    callbacks=[TQDMNotebookCallback(leave_outer=True, leave_inner=True)])


model01.save('model01.pkl')