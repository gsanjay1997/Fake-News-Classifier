# -*- coding: utf-8 -*-
"""Fake News Classifier - Sanjay Govindarajan"""

#Importing the packages
import pandas as pd
import numpy as np
import string
#Natural Language Toolkit Package to access the Commonly used words
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
#For counting purpose
from collections import Counter
#For special character removal purpose
import re
#STEMMING
from nltk.stem.porter import PorterStemmer
#LEMMATIZATION & POS TAG
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
#Spell checker
from spellchecker import SpellChecker
#Model Training
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
#Pickling
import pickle

def data_cleaning(data):
  data_label_count = data['label'].value_counts()
  #title_text_null = data[(data['text'].isna() == True) & (data['title'].isna() == True)]
  label_null = data[(data['label'].isna() == True)]
  #data.drop(title_text_null.index, inplace = True)
  data.drop(label_null.index, inplace = True)
  data.reset_index(drop = True, inplace = True)
  data = data.replace(to_replace = np.nan, value = "na")
  return data_preprocessing(data)

def data_preprocessing(data):
  word_counts = Counter()
  punc = string.punctuation
  stop_words = set(stopwords.words('english'))
  stop_words.remove('no')
  for column in ['title', 'text']:
    data[column] = data[column].apply(lambda x : x.lower() if isinstance(x, str) else x)
    data[column] = data[column].apply(lambda x : x.translate(str.maketrans('', '', punc)))
    data[column] = data[column].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    data[column] = data[column].apply(lambda x : re.sub('\s+', ' ', x))
    data[column] = data[column].apply(lambda x : re.sub(r'<.*?>', '', x))
    data[column] = data[column].apply(lambda x : re.sub(r'https?://\S+|www\.\S+', '', x))
    data[column] = data[column].apply(lambda x : re.sub('[^a-z0-9]', ' ', x))
    for data_sentence in data[column]:
      for word in data_sentence.split():
        word_counts[word] += 1
  word_top_list = set(word for (word, label) in word_counts.most_common()[:51])
  word_bottom_list = set(word for (word, label) in word_counts.most_common()[:-101:-1])
  remove_word = {'like', 'also', 'would', 'mr', 'could', 'said', 'us', 't', 's', 'i', 'going', 'may', 'get', 'say', 'many', 'it'}
  remove_word.update(word_bottom_list)
  data[column] = data[column].apply(lambda x: ' '.join([word for word in x.split() if word not in remove_word]))
  print(f'Top 20 frequently used words: {word_top_list}')
  print(f'Top 100 least words: {word_bottom_list}')
  print(f'Words removed from the model: {remove_word}')
  return data

def spell_check(data):
  spell = SpellChecker()
  wrng_wrd = []
  for column in ['title', 'text']:
    for data_sentence in data[column]:
      wrng_wrd = spell.unknown(str(data_sentence.split()))
    data[column] = data[column].apply(lambda x: " ".join([spell.correction(word) if word in wrng_wrd else word for word in x.split()]))
  print(f'Wrong words: {wrng_wrd}')
  return data

def stem_words(data):
  ps = PorterStemmer()
  for col in ['title', 'text']:
    data[col] = data[col].apply(lambda x: ' '.join([ps.stem(word) for word in x.split()]))
  return data

def lemmatize_words(data):
  lem = WordNetLemmatizer()
  wordmap = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
  for col in ['title', 'text']:
    data[col] = data[col].apply(lambda x: ' '.join([lem.lemmatize(word, wordmap.get(pos[0], wordnet.NOUN)) for word, pos in pos_tag(x.split())]))
  return data

def fake_news_detection(input):
  input = input.replace(to_replace = np.nan, value = "na")
  #input = stem_words(input)
  input = lemmatize_words(input)
  input = spell_check(input)
  input = input[['title', 'text']].apply(lambda row: ' '.join(row), axis=1)
  input_vector = vector_form.transform(input)
  prediction = load_model.predict(input_vector)
  return prediction

# We are importing the FAKE NEWS CLASSIFIER
# Label "1" is considered as REAL news
# Label "0" is considered as FAKE news
col = ['title', 'text', 'label']
data = pd.read_csv("/content/drive/MyDrive/NEWS CLASSIFIER/Dataset.csv", usecols = col)
data = data_cleaning(data)
data = data_preprocessing(data)
#data = stem_words(data)
data = lemmatize_words(data)
data = spell_check(data)
x = data[['title', 'text']].apply(lambda row: ' '.join(row), axis=1)
y = data['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
vect = TfidfVectorizer()
x_train = vect.fit_transform(x_train)
x_test = vect.transform(x_test)
model = DecisionTreeClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(f'Accuracy: {model.score(x_test, y_test)}')
pickle.dump(vect, open('vector.pkl', 'wb'))
pickle.dump(model, open('model.pkl', 'wb'))
vector_form=pickle.load(open('vector.pkl', 'rb'))
load_model=pickle.load(open('model.pkl', 'rb'))
print('Enter the tite of NEWS:\n')
in_title = input()
print('Enter the text of NEWS:\n')
in_text = input()
input_df = pd.DataFrame([[in_title, in_text]], columns = ['title', 'text'])
output = fake_news_detection(input_df)
if output == 1:
  print('The NEWS is REAL')
else:
  print('The NEWS is FAKE')
