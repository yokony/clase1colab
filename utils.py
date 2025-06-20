import nltk
nltk.download('punkt_tab')
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

import random
import json

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense

from sklearn.preprocessing import LabelEncoder

# Cargar los datos
path = "/content/drive/MyDrive/GRUPO_166/CHATBOT/intents.json"
with open(path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# creamos el stemmer
stemmer = PorterStemmer()

# Preprocesamiento (vocab = vocabulario)
vocab = []
tags = []
patterns = []
labels = []

for intent in data['intents']:
    for Pattern in intent['patterns']:
        tokens = word_tokenize(Pattern.lower())
        stemmed = [stemmer.stem(w) for w in tokens]
        vocab.extend(stemmed) # llena en lista (varias palabras)
        patterns.append(stemmed) # llena de una en una
        labels.append(intent['tag'])
    if intent['tag'] not in tags:
        tags.append(intent['tag'])

vocab = sorted(set(vocab))

#one4 -hot input
X= []
Y= []

encoder = LabelEncoder()
encoder_labels = encoder.fit_transform(labels)

for pattern in patterns:
    bag = [1 if word in pattern else 0 for word in vocab]
    X.append(bag)

Y = encoder_labels 
# convertimos las variables a arreglos de numpy
X = np.array(X)
Y = np.array(Y)


#Modelo
D = len(X[0]) #cantidad de entradas
C = len(tags) # cantidad de etiquetas

model = Sequential()
# capa de entrada - Densa
model.add(Dense(8, input_shape= (D,), activation = 'relu'))
# capa densa 2
model.add(Dense(8, activation = 'relu'))
model.add(Dense(C, activation = 'softmax'))

# compilamos 
model.compile(loss='sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

#model.compile(
 #   loss='sparse_categorical_crossentropy',
  #  optimizer = 'adam',
   # metrics = ['accuracy']
#)
# entrenamos

model.fit(X, Y, epochs=200, verbose=0)

# Función para procesar la entrada
def predict_class(text):
    tokens = word_tokenize(text.lower())
    stemmed = [stemmer.stem(w) for w in tokens]
    bag = np.array([1 if word in stemmed else 0 for word in vocab])
    res = model.predict(np.array([bag]), verbose = 0)[0]
    idx = np.argmax(res)
    tag = encoder.inverse_transform([idx])[0]
    return tag

#Función para dar las respuestas
def get_response(tag, context):
    for intent in data['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "No enendí eso, ¿puedes repetirlo?"






