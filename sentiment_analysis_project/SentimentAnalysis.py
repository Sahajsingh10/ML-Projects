import numpy as np
from textblob import TextBlob, blob
import tweepy
from textblob.classifiers import NaiveBayesClassifier
from textblob.sentiments import NaiveBayesAnalyzer
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import pandas as pd



api_key = 'XXXXXXXXXXXXXXXXXXXXXX'
api_secret = 'XXXXXXXXXXXXXXXXXXXXXXX'
access_token = 'XXXXXXXXXXXXXXXXXXXXXX'
access_secret = 'XXXXXXXXXXXXXXXXXXXXXXXXX'

twitter = tweepy.OAuthHandler(api_key, api_secret)
api = tweepy.API(twitter)
pres_Biden = ""
corpus_tweets = api.search("Bipartisan,Infrastructure, 70, years", count = 1, lan = "en", type = 'recent')
for tweet in corpus_tweets:
    pres_Biden = pres_Biden + tweet.text
print("President Biden")
print(pres_Biden)

blob_object = TextBlob(tweet.text, analyzer=NaiveBayesAnalyzer())
analysis1 = blob_object.subjectivity
print("subjectivity", analysis1)
analysis2 = TextBlob(pres_Biden).polarity
print("Polarity", analysis2)

train = [
    ('These are significant', 'pos'),
    ('lower costs', 'pos'),
("I can't deal with this", 'neg')
]
test = [
    ('These are significant costs', 'pos'),
  ("I can't believe I'm doing this.", 'neg'),
    ('it will pass it', 'pos'),
    ('it will BEUTIFULLY pass it', 'pos')


]


print()
print('Classification:')
cl = NaiveBayesClassifier(train)
blob = TextBlob(pres_Biden, classifier=cl)
blob.classify()
for sentence in blob.sentences:
     print(sentence)
     print(sentence.classify())
cl.accuracy(test)
cl.show_informative_features(5)
wordcloud = WordCloud(max_font_size=50, max_words=20, background_color="black").generate(pres_Biden)
plt.figure()
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
print()
print()
print()
print("Lebron James")
lebron = ""
corpus_tweets = api.search("BEAUTIFULLY, GORGEOUS, man", count = 1, lan = "en", type = 'recent')
for i in corpus_tweets:
    lebron = lebron + i.text

print(lebron)
blob_object2 = TextBlob(tweet.text, analyzer=NaiveBayesAnalyzer())
analysis2 = blob_object2.subjectivity
print("subjectivity", analysis2)
analysis3 = TextBlob(lebron).polarity
print("Polarity", analysis3)
print()
print('Classification:')
tr = [
    ("They are nice", "pos"),
    ("so good man!", "pos"),
    ("They are so bad", 'neg')
]
te = [
    ('These are nice', 'pos'),
  ("I love this man!", 'pos'),

]

clt = NaiveBayesClassifier(tr)
blob1 = TextBlob(lebron, classifier=clt)
blob1.classify()
for j in blob1.sentences:
     print(j)
     print(j.classify())
clt.accuracy(te)
clt.show_informative_features(5)
wordcloud = WordCloud(max_font_size=50, max_words=20, background_color="black").generate(lebron)
plt.figure()
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
#graph

x = ['GroupA', 'GroupB']
PresBiden =[0.5416, 0.2083]
lebronJ = [0.5416, 1.0]
x_axis = np.arange(len(x))

plt.bar(x_axis - 0.2, PresBiden, 0.4, label = 'President Biden')
plt.bar(x_axis + 0.2, lebronJ, 0.4, label = 'Lebron James')
plt.xlabel('Polarity and Subjectivity')
plt.xticks(x_axis, x)
plt.xlabel("Polarity and Subjectivity")
plt.ylabel("Number")
plt.title("President Biden and Lebron James tweet sentiments")
plt.legend()
plt.show()