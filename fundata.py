from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from selenium.webdriver.chrome.options import Options
import re
import mysql.connector
import pandas as pd
import sys
import math
def my_function():
  print("Hello from a function")
def scraprevs(restname,exitflag):
  print(exitflag)
  options = Options()
  options.add_experimental_option('detach', True)
  driver = webdriver.Chrome('./chromedriver',options=options)
  driver.maximize_window()
  driver.get("https://www.google.com/")
  print(driver.title)
  buttons = driver.find_elements_by_xpath("//*[contains(text(), 'Alle akzeptieren')]")

  for btn in buttons:
      btn.click()
  time.sleep(1)
  print(driver.title)
  p=driver.find_element_by_name("q")
  p.clear()
  p.send_keys(restname)
  p.send_keys(Keys.RETURN)
  time.sleep(1)
  print(driver.title)
  driver.find_element_by_link_text('Change to English').click()
  time.sleep(1)
  print(driver.title)
  nrev=int(driver.find_element_by_partial_link_text('Google-Rezensionen').text.split(" ")[0])
  driver.find_element_by_partial_link_text('Google-Rezensionen').click()
  time.sleep(1)
  print(math.floor(nrev/4))
  print(driver.title)
  #driver.find_elements_by_class_name("review-dialog-list").execute_script("window.scrollTo(0, 10);")
  list1 = driver.find_elements_by_class_name("Jtu6Td")
  listlen1=len(list1)
  listtext=[]
  for i in range(math.floor(nrev/2)):
          driver.execute_script("arguments[0].scrollIntoView();", list1[i*4])
          time.sleep(1)
          list1 = driver.find_elements_by_class_name("Jtu6Td")
  for item in driver.find_elements_by_link_text('More'):
          item.click()
  time.sleep(4)
  list1 = driver.find_elements_by_class_name("Jtu6Td")
  index=0
  for item in list1:
    if (item.text != ''):
      if(len(item.text.split('(Translated by Google)'))==1):
        listtext.append(item.text)
        #print([item.text, index])
        index=index+1
      else:
        listtext.append(item.text.split('(Translated by Google)')[1].split('(Original)')[0])
        index = index + 1
  #print(list1[2].text)
  time.sleep(1)
  list2 = driver.find_elements_by_class_name("TSUbDb")
  listnames = []
  index = 0
  for item in list2:
    listnames.append(item.text)
    # print([item.text, index])
    index = index + 1
  # print(listtext)
  # print(exitflag==1)

  if exitflag==1:
          driver.close()
  return [listnames,listtext]



def makesqltab(list1,list2):
  mydb = mysql.connector.connect(
    host="localhost",
    user="shashank",
    password="Suj@2305",
    database="mydatabase",
    charset = "utf8"
  )
  relist1=[]
  for item in list1:
    relist1.append(re.escape(item))
  relist2 = []
  for item in list2:
    relist2.append(re.escape(item))

  try:
    mycursor = mydb.cursor()

    mycursor.execute("CREATE TABLE mydatabase.reviewtab (name BLOB, reviews BLOB)")
  except:
    mycursor = mydb.cursor()

    sql = "DROP TABLE reviewtab"

    mycursor.execute(sql)

    mycursor = mydb.cursor()

    mycursor.execute("CREATE TABLE mydatabase.reviewtab (name BLOB, reviews BLOB)")

  mycursor = mydb.cursor()

  mycursor.execute("SHOW TABLES")

  for x in mycursor:
    print(x)

  mycursor = mydb.cursor()

  mycursor.execute("ALTER TABLE reviewtab ADD COLUMN id INT AUTO_INCREMENT PRIMARY KEY FIRST")

  # list1=["one","one","one","one","one"]
  # list2=list1
  val=list(zip(list1,list2))
  # print(val[11])
  # debugele=5
  # print(debugele)
  # print(*relist1[0:debugele],sep='\n----\n')
  # print(relist2[0:debugele])

  # print(val[12])
  # print(val[13])

  mycursor = mydb.cursor()

  sql = "INSERT INTO reviewtab (name, reviews) VALUES (%s, %s)"
  # val = [
  #   ('Peter', 'Lowstreet 4'),
  #   ('Amy', 'Apple st 652'),
  #   ('Hannah', 'Mountain 21'),
  #   ('Michael', 'Valley 345'),
  #   ('Sandy', 'Ocean blvd 2'),
  #   ('Betty', 'Green Grass 1'),
  #   ('Richard', 'Sky st 331'),
  #   ('Susan', 'One way 98'),
  #   ('Vicky', 'Yellow Garden 2'),
  #   ('Ben', 'Park Lane 38'),
  #   ('William', 'Central st 954'),
  #   ('Chuck', 'Main Road 989'),
  #   ('Viola', 'Sideway 1633')
  # ]

  mycursor.executemany(sql, val)

  mydb.commit()

  print(mycursor.rowcount, "was inserted.")

  mycursor = mydb.cursor()

  mycursor.execute("SELECT * FROM reviewtab")

  myresult = mycursor.fetchall()

  for x in myresult:
    print(x)

  # mycursor = mydb.cursor()
  #
  # sql = "DROP TABLE reviewtab"
  #
  # mycursor.execute(sql)





def getsqltab():
  mydb = mysql.connector.connect(
    host="localhost",
    user="shashank",
    password="Suj@2305",
    database="mydatabase",
    charset="utf8"
  )
  c = mydb.cursor()

  c.execute('''
            SELECT
            *
            FROM reviewtab
            ''')

  df = pd.DataFrame(c.fetchall(), columns=['id', 'name', 'reviews'])
  # print(df)
  # df= df.name.decode(encoding = 'UTF-8')
  namelst=df["name"].str.decode(encoding = 'UTF-8')
  revlist=df["reviews"].str.decode(encoding='UTF-8')
  # print(namelst)
  # print(revlist)
  # print(df._get_value(1, "name").decode('UTF-8'))
  return [namelst,revlist]

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import *


p_stemmer = PorterStemmer()





def nltk_process(text):

    words = set(nltk.corpus.words.words())

    sent = text
    test1=" ".join(w for w in nltk.wordpunct_tokenize(sent) \
             if w.lower() in words or not w.isalpha())

    # Tokenization
    nltk_tokenList = word_tokenize(test1)

    # Stemming
    nltk_stemedList = []
    for word in nltk_tokenList:
        nltk_stemedList.append(p_stemmer.stem(word))

    # Lemmatization
    wordnet_lemmatizer = WordNetLemmatizer()
    nltk_lemmaList = []
    for word in nltk_stemedList:
        nltk_lemmaList.append(wordnet_lemmatizer.lemmatize(word))

    # print("Stemming + Lemmatization")
    # print(nltk_lemmaList)
    # Filter stopword
    filtered_sentence = []
    nltk_stop_words = set(stopwords.words("english"))
    for w in nltk_lemmaList:
        if w not in nltk_stop_words:
            filtered_sentence.append(w)
    # Removing Punctuation
    punctuations = "?:!.,;"
    for word in filtered_sentence:
        if word in punctuations:
            filtered_sentence.remove(word)
    # print(" ")
    # print("Remove stopword & Punctuation")
    # print(filtered_sentence)
    return " ".join(filtered_sentence)


def adject(text1):
    # get adjectives
    tagstext2 = []
    tagstext = nltk.pos_tag(word_tokenize(text1))
    for (item1, item2) in tagstext:
        if item2 == 'JJ':
            tagstext2.append(item1)
    adjassent = " ".join(tagstext2)
    return adjassent

