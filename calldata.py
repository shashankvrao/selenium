from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import sys
import re
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from fundata import *

from PIL import ImageFont, ImageDraw


my_function()
# [names,reviews]=scraprevs("CAPOBIANCO essen",1)
# [names,reviews]=scraprevs("mausefalle mülheim",0)
# [names,reviews]=scraprevs("pizzeria margherita mülheim",1)

# makesqltab(names,reviews)

[names1,reviews1]=getsqltab()

text1 = ' '.join(reviews1)
text3=nltk_process(text1)
text2 = adject(text3)

# print(text2)


# Create and generate a word cloud image:
wordcloud = WordCloud().generate(text2)
#
# Display the generated image:
# plt.imshow(wordcloud)
# plt.axis("off")
# plt.show()
import _pickle as cPickle
# load it again
from sklearn.feature_extraction.text import CountVectorizer
with open('my_dumped_cv.pkl', 'rb') as fid:
    cv = cPickle.load(fid)

# load it again
from sklearn.naive_bayes import MultinomialNB
with open('my_dumped_classifier.pkl', 'rb') as fid:
    classifier = cPickle.load(fid)

import numpy as np
import pandas as pd
# Importing the dataset
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, len(reviews1)):
    reviewt = re.sub('[^a-zA-Z]', ' ', reviews1[i])
    reviewt = reviewt.lower()
    reviewt = reviewt.split()
    ps = PorterStemmer()
    reviewt = [ps.stem(word) for word in reviewt if not word in set(stopwords.words('english'))]
    reviewt = ' '.join(reviewt)
    corpus.append(reviewt)



A=cv.transform(corpus)
# print(A)
A_pred = classifier.predict(A)
predict=list(zip(reviews1,A_pred))
# itemnum=2
# print(predict[itemnum][0])
# print(predict[itemnum][1])
import csv

with open('reviewsqual.csv', 'w') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerows(predict)

from statistics import mean
print(np.average(A_pred))
vegrevs=[]
vegrevsnam=[]
vegrevind=[]
for index, item in enumerate(reviews1, start=0):   # default is zero
    if "vegetarian" in item:
        vegrevs.append(item)
        vegrevsnam.append(names1[index])
        vegrevind.append(index)
    if "Vegetarian" in item:
        vegrevs.append(item)
        vegrevsnam.append(names1[index])
        vegrevind.append(index)
    if "VEGETARIAN" in item:
        vegrevs.append(item)
        vegrevsnam.append(names1[index])
        vegrevind.append(index)
    if "vegan" in item:
        vegrevs.append(item)
        vegrevsnam.append(names1[index])
        vegrevind.append(index)
    if "Vegan" in item:
        vegrevs.append(item)
        vegrevsnam.append(names1[index])
        vegrevind.append(index)
    if "VEGAN" in item:
        vegrevs.append(item)
        vegrevsnam.append(names1[index])
        vegrevind.append(index)

corpus = []
for i in range(0, len(vegrevs)):
    reviewt = re.sub('[^a-zA-Z]', ' ', vegrevs[i])
    reviewt = reviewt.lower()
    reviewt = reviewt.split()
    ps = PorterStemmer()
    reviewt = [ps.stem(word) for word in reviewt if not word in set(stopwords.words('english'))]
    reviewt = ' '.join(reviewt)
    corpus.append(reviewt)



A=cv.transform(corpus)
# print(A)
A_pred = classifier.predict(A)
predict=list(zip(reviews1,A_pred))
# itemnum=2
# print(predict[itemnum][0])
# print(predict[itemnum][1])
import csv

with open('vegreviewsqual.csv', 'w') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerows(predict)

from statistics import mean
print(np.average(A_pred))
print(len(reviews1))
print(A_pred)
print(vegrevs[0])

