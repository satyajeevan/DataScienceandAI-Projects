#IMDB reviews - Interstellar Movie
library(rvest)
library(XML)
library(magrittr)
#IMDB Interstellar movie
imdb_aurl="https://www.imdb.com/title/tt0816692/reviews?ref_=tturv_ql_3"

imdb_reviews <- NULL
## Web scraping- "Page source"
#Under the "text" we can find the content

murl <- read_html(as.character(imdb_aurl))
rev <- murl %>% html_nodes(".text") %>% html_text()
imdb_reviews <- c(imdb_reviews,rev)

#Copy the data 
txt_imdb <- imdb_reviews

#corpus 
x <- Corpus(VectorSource(txt_imdb))
x <- tm_map(x, function(x) iconv(enc2utf8(x), sub='byte'))

# Data Cleansing
#Text into lower case
x1 <- tm_map(x,tolower)
#Removing the punctuations in the text
x1 <- tm_map(x1,removePunctuation)
#removing the numbers
x1 <- tm_map(x1,removeNumbers)
#removing english stop words- i.e a,an,the,pronouns
stopwords <-  readLines(file.choose())
x1 <- tm_map(x1,removeWords,stopwords("English"))
# striping white spaces 
x1 <- tm_map(x1,stripWhitespace)
inspect(x1[1])

# Term document matrix 
# converting unstructured data to structured format using TDM

tdm <- TermDocumentMatrix(x1)
tdm
dtm <- t(tdm) # transpose
dtm <- DocumentTermMatrix(x1)

# To remove sparse entries upon a specific value - which are empty
corpus.dtm.frequent <- removeSparseTerms(tdm, 0.99) 

tdm <- as.matrix(tdm)
dim(tdm)

# Create term document matrices from raw text 
tdm[1:1731, 1:25]

w <- rowSums(tdm)
w

#Generate the bar plot for observing the word frequency

w_sub <- subset(w, w>=10)
w_sub

barplot(w_sub, las=2, col = rainbow(10))


# "movie","cinema","film","say","us","interstellar" repeats maximum number of times
x1 <- tm_map(x1, removeWords, c("movie","just","will","like","can","cinema","film","say","us","interstellar","way","see","nolan"))
x1 <- tm_map(x1, stripWhitespace)

tdm <- TermDocumentMatrix(x1)
dim(tdm)


#After removal of stop words
tdm <- as.matrix(tdm)
tdm[1:1723, 1:25]

w <- rowSums(tdm)
w

w_sub <- subset(w, w>=10)
w_sub

barplot(w_sub, las=2, col = rainbow(10))
######################################################

library(wordcloud)

wordcloud(words = names(w), freq = w)

w_sub1 <- sort(rowSums(tdm), decreasing = TRUE)
head(w_sub1)

wordcloud(words = names(w_sub1), freq = w_sub1) # all words are considered

# better visualization
wordcloud(words = names(w_sub1), freq = w_sub1, random.order=F, colors= rainbow(30),scale=c(3,0.5),rot.per=0.3)

# Wordcloud2 - shapes of circle,triangle,star

installed.packages("wordcloud2")
library(wordcloud2)

w1 <- data.frame(names(w_sub),w_sub)
colnames(w1) <- c('word', 'freq')
#special type of word cloud
wordcloud2(w1, size=0.3, shape = 'star')

#####################################
# lOADING Positive and Negative words  
pos.words <- readLines(file.choose())	# read-in positive-words.txt
neg.words <- readLines(file.choose()) 	# read-in negative-words.txt

#Positive word cloud - IMDB Insterstellar
pos.matches <- match(names(w), pos.words)
pos.matches <- !is.na(pos.matches)
freq_pos <- w[pos.matches]
names <- names(freq_pos)
windows()
wordcloud(names, freq_pos, scale=c(4,1), colors = brewer.pal(8,"Dark2"))


#Negative word cloud - IMDB Insterstellar
neg.matches <- match(names(w), neg.words)
neg.matches <- !is.na(neg.matches)
freq_neg <- w[neg.matches]
names <- names(freq_neg)
windows()
wordcloud(names, freq_neg, scale=c(4,.5), colors = brewer.pal(8, "Dark2"))


#Bigram Anlysis on Interstellar movie
library(RWeka)
library(wordcloud)

minfreq_bigram <- 2
bitoken <- NGramTokenizer(x1, Weka_control(min = 2, max = 2))
two_word <- data.frame(table(bitoken))
sort_two <- two_word[order(two_word$Freq, decreasing = TRUE), ]

wordcloud(sort_two$bitoken, sort_two$Freq, random.order = F, scale = c(2, 0.35), min.freq = minfreq_bigram, colors = brewer.pal(8, "Dark2"), max.words = 150)

