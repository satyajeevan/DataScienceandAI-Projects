# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 15:41:33 2020

@author: Jeevan kumar
"""

import requests
from bs4 import BeautifulSoup as bs # web scraping
import re # regular expression

new=[]
imdb_reviews=[]

# Web scraping using beautiful soup - “Page source”
from wordcloud import WordCloud
#URL
url="https://www.imdb.com/title/tt0816692/reviews?ref_=tturv_ql_3"
response=requests.get(url)
soup=bs(response.content,"html.parser")
#Extracting by html.parser and soup under class  “div”  & text show-more__control under this we can find the review of the user
imdb_reviews=soup.find_all("div",attrs={"class","text show-more__control"})
for i in range(15):
    k=imdb_reviews[i] 
    new+=k.text

Interstellar="".join(new)

import nltk


Interstellar= re.sub("[^A-Za-z" "]+"," ", Interstellar).lower()
Interstellar= re.sub("[0-9" "]+"," ", Interstellar)
Interstellar

# writng reviews in a text file 
with open("interstellar review.txt","w",encoding='utf8') as output:
    output.write(str(Interstellar))
    
Review_split_Insterstellar=Interstellar.split()

from nltk.corpus import stopwords

#TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(Review_split_Insterstellar, use_idf=True,ngram_range=(1, 3))
X = vectorizer.fit_transform(Review_split_Insterstellar)

stop_words = stopwords.words('English')
Review_split_Insterstellar = [w for w in Review_split_Insterstellar if not w in stop_words]
Interstellar_RmStop=" ".join(Review_split_Insterstellar)


# WordCloud can be performed on the string inputs.
# Corpus level word cloud for ipad mini reviews
import matplotlib.pyplot as plt
wordcloud_imdb_reviews = WordCloud(background_color='White',width=1800,height=1400).generate(Interstellar_RmStop)
plt.imshow(wordcloud_imdb_reviews)

stop_words.extend(["movie","cinema","film","say","us","interstellar","way","see","nolan"])

Review_split_Insterstellar = [w for w in Review_split_Insterstellar if not w in stop_words]
Interstellar_RmStop=" ".join(Review_split_Insterstellar)

#After Removal of stop words
wordcloud_imdb_reviews = WordCloud(background_color='black',width=1800,height=1400).generate(Interstellar_RmStop)
plt.title("Removal of stop words ")
plt.imshow(wordcloud_imdb_reviews)


# Positive word cloud
# positive words # Choose the path for +ve words stored in system
with open("J:\\DataScienceAndAI\\Text_mining\\assign\\positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")
  
# Choosing the only words which are present in positive words
imdb_in_pos = " ".join ([w for w in Review_split_Insterstellar if w in poswords])

wordcloud_imdb_pos = WordCloud(background_color='Black',width=1800,height=1400).generate(imdb_in_pos)
plt.title("Interstellar Movie +Positive word cloud- Imdb Review")
plt.imshow(wordcloud_imdb_pos)


with open("J:\\DataScienceAndAI\\Text_mining\\assign\\negative-words.txt","r") as pos:
  neg_word = pos.read().split("\n")

# Choosing the only words which are present in negative words
imdb_in_neg = " ".join ([w for w in Review_split_Insterstellar if w in neg_word])

wordcloud_imdb_neg = WordCloud(background_color='white',width=1800,height=1400).generate(imdb_in_neg)
plt.title("Interstellar Movie ~Negative word cloud- Imdb Review")
plt.imshow(wordcloud_imdb_neg)

# Biagram

Imdb_lemm = nltk.WordNetLemmatizer()

# Lowercase and tokenize
New_text = Interstellar_RmStop.lower()

# Remove single quote early since it causes problems with the tokenizer.
New_text = New_text.replace("'", "")

tokens_imdb = nltk.word_tokenize(New_text)
New_text_2 = nltk.Text(tokens_imdb)

# Remove extra chars and remove stop words.
Final_content = [''.join(re.split("[ .,;:!?‘’``''@#$%^_&*()<>{}~\n\t\\\-]", word)) for word in New_text_2]

# Take only non-empty entries
Final_content = [s for s in Final_content if len(s) != 0]

# Best to get the lemmas of each word to reduce the number of similar words
Final_content = [Imdb_lemm.lemmatize(t) for t in Final_content]

nltk_tokens_imdb = nltk.word_tokenize(New_text)  
bigrams_list_new = list(nltk.bigrams(Final_content))
print(bigrams_list_new)

dictionary_imdb_biagram = [' '.join(tup1) for tup1 in bigrams_list_new]
print (dictionary_imdb_biagram)

# Using count vectoriser to view the frequency of bigrams
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range=(2, 2))
bag_of_words = vectorizer.fit_transform(dictionary_imdb_biagram)
vectorizer.vocabulary_

sum_words_imdb= bag_of_words.sum(axis=0)
words_freq_imdb = [(word, sum_words_imdb[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq_imdb =sorted(words_freq_imdb, key = lambda x: x[1], reverse=True)
print(words_freq_imdb[:100])

# Generating IMDB_Bigram_wordcloud
words_dict_imdbb = dict(words_freq_imdb) ;WC_height = 1000;WC_width = 1500 ;WC_max_words = 200
WordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width)
WordCloud.generate_from_frequencies(words_dict_imdbb)

#Bigrams in the plot stimulates the same color representing comes under single word instead of two different words
plt.title('Bigrams word cloud -Most frequently occurring are connected by same colour and font size')
plt.imshow(WordCloud, interpolation='bilinear')
plt.axis("off")
plt.show()

