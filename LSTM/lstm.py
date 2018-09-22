# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 22:42:30 2018

@author: Nidhi

"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf #magical gradient computation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nltk
from nltk.corpus import stopwords
import  json
nltk.download('punkt')


heading,body,label=[],[],[]
for i in range(1,121):
    
   path="C:/Users/Nidhi/Desktop/FakeNewsNet-master/FakeNewsNet-master/Data/PolitiFact/FakeNewsContent/PolitiFact_Fake_"+str(i)+"-Webpage.json"

   fake_file=json.loads(open(path).read())
   
   body.append(fake_file['text'])
   heading.append(fake_file['title'])
   label.append(0)
   

for i in range(1,121):
    
   path="C:/Users/Nidhi/Desktop/FakeNewsNet-master/FakeNewsNet-master/Data/PolitiFact/RealNewsContent/PolitiFact_Real_"+str(i)+"-Webpage.json"

   Real_file=json.loads(open(path).read())
   
   body.append(Real_file['text'])
   heading.append(Real_file['title'])
   label.append(1)
   
df=pd.DataFrame()
df['heading']=heading
df['body']=body
df['label']=label
df['length']=[len(heading) for heading in df['heading']]
df['length'].describe()
real_text = ' '.join(df[df['label'] == 1]['heading'])
fake_text = ' '.join(df[df['label'] == 0]['heading'])
fake_words = [word for word in nltk.tokenize.word_tokenize(fake_text) if word not in stopwords.words('english') and len(word) > 3]
real_words = [word for word in nltk.tokenize.word_tokenize(real_text) if word not in stopwords.words('english') and len(word) > 3]

common_fake = nltk.FreqDist(fake_words).most_common(25)
common_real =nltk.FreqDist(real_words).most_common(25)
fake_ranks = []
fake_counts = []
real_ranks = []
real_counts = []

for ii, word in enumerate(reversed(common_fake)):
    fake_ranks.append(ii)
    fake_counts.append(word[1])

for ii, word in enumerate(reversed(common_real)):
    real_ranks.append(ii)
    real_counts.append(word[1])

plt.figure(figsize=(20, 7))

plt.scatter(fake_ranks, fake_counts)

for labels, fake_rank, fake_count in zip(common_fake, fake_ranks, fake_counts):
    plt.annotate(
        labels[0],
        xy = (fake_rank, fake_count)
    )
    
    plt.scatter(real_ranks, real_counts)
plt.title('Real vs Fake Headlines')

for labels, real_rank, real_count in zip(common_real, real_ranks, real_counts):
    plt.annotate(
        labels[0],
        xy = (real_rank, real_count)
    )
    
real_patch = mpatches.Patch(color='orange', label='Real')
fake_patch = mpatches.Patch(color='blue', label='Fake')
plt.legend(handles=[real_patch, fake_patch])

plt.show()
def pad(x):
    
    if len(x) < 69:
        
        return x + ' ' * (69 - len(x))
    
    return x

def trim(x):
    
    if len(x) > 69:
        
        return x[:69]
    
    return x

df['heading'] = df['heading'].apply(pad)
df['heading'] = df['heading'].apply(trim)
df['length'] = [len(heading) for heading in df['heading']]
df.describe()
text = ' '.join(df['heading'])
dictionary_size = len(set(text))
dictionary = sorted(set(text))
character_map = { k:v for v, k in enumerate(dictionary) }
max_length = 69
batch_size = 50
def to_input(sentence, character_map, dictionary_size):
    
    sentence = np.array([character_map[char] for char in sentence])
    one_hot = np.zeros((len(sentence), dictionary_size))
    one_hot[np.arange(len(sentence)), sentence] = 1
    return one_hot
def batch(sentences, labels, start, batch_size):
    
    if start + batch_size < len(sentences):
        
        inputs = [to_input(sentence, character_map, dictionary_size) for sentence in sentences[start: start + batch_size ]]
        labels = [label for label in labels[start: start + batch_size ]]
        start = start + batch_size
    
    else:
        
        inputs = [to_input(sentence, character_map, dictionary_size) for sentence in sentences[start:]]
        labels = [label for label in labels[start:]]
        start = 0
    
    return np.array(inputs), np.array(labels) , start
def test_batch(sentences, labels):
    
        
    inputs = [to_input(sentence, character_map, dictionary_size) for sentence in sentences]
    labels = [label for label in labels]

    return np.array(inputs), np.array(labels)