# import spacy
# from spacy.cli.download import download
# download(model="en_core_web_sm")
#
# nlp = spacy.blank("en")
# doc = nlp("This is a sentence.")

# from chatterbot import ChatBot
# from chatterbot.trainers import ListTrainer
#
# chatbot = ChatBot('Flash')
# trainer = ListTrainer(chatbot)
# trainer.train([
#     "Hi, can I help you?",
#     "Sure, I'd like to book a flight to San-Francisco.",
#     "Your flight has been booked.The Airline is Emirates"
# ])
# response = chatbot.get_response('I would like to book a flight.')
# print(response)


import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
# load the file in reading binary

model = load_model('chatbotmodel.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i,word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    result=[[i, r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]

    result.sort(key=lambda x: x[1], reverse=True)
    return_list = []

    for r in result:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print("GO! BOT IS RUNNING !")
while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)
