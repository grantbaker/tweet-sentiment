import gensim
import numpy as np
import json
from sklearn import svm
import re
from collections import namedtuple
from random import shuffle
from copy import copy
import ijson
import os
import pickle as pkl

filename = 'moreTweets.json'
'''
tweets = []
with open(filename, 'r') as f:
    #tweets = json.load(f)
    objects = ijson.items(f, 'item')
    i = 0
    for row in objects:
        #print(row)
        if i==0:
            tweet = {}
            tweet['text'] = row['tweet']['text']
            tweet['lang'] = row['tweet']['lang']
            #tweet['text'] = row['tweet']['text']
            i=1
        else:
            tweet['sentiment'] = row['sentiment']
            if tweet['lang']=='en':
                tweets.append(tweet)
            i=0

        #good_columns=['tweet']
        #for item in good_columns:
        #    selected_row.append(row[column_names.index(item)])
        #data.append(selected_row)
#print(tweets)
#tweets = json.load(f)
#tweets_fixed = []

#for i in range(int(len(tweets)/2)):
#    if tweets[2*i]['tweet']['lang'] == 'en':
#        tweets[2*i]['sentiment'] = tweets[2*i+1]['sentiment']
#        tweets[2*i]['used'] = 1
#        tweets_fixed.append(tweets[2*i])
#tweets = tweets_fixed
print(np.shape(tweets))

p=open('tweets.pkl', 'wb')
pkl.dump(tweets,p)
p.close()'''

p=open('tweets.pkl', 'rb')
tweets=pkl.load(p)
p.close()
for tweet in tweets:
    tweet['used'] = 1
#tweets = tweets[0:100000]

print('Loaded Tweets')
print(tweets[0]['sentiment']['SentimentScore']['Positive']>0.5)
tweetTexts = []
analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
for i, tweet in enumerate(tweets):
    text = tweet['text']
    tags = [i]
    words = re.findall(r"[\w']+", text)
    tweet['docTag'] = tags
    tweetTexts.append(analyzedDocument(words, tags))
    #print(words)
tweetTexts = list(tweetTexts)

print('tweetTexts')
'''
docModel = gensim.models.doc2vec.Doc2Vec(min_count=1)
docModel.build_vocab(tweetTexts)
a = copy(tweetTexts)
for i in range(10):
    shuffle(a)
    docModel.train(a, total_examples = len(tweetTexts), epochs = 1)
    print('Epoch: ', i)

p=open('docModel.pkl', 'wb')
pkl.dump(docModel,p)
p.close()
'''
p=open('docModel.pkl', 'rb')
docModel=pkl.load(p)
p.close()
print('Doc Model Loaded')
'''
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
p=open('model.pkl', 'wb')
pkl.dump(model,p)
p.close()
'''

p=open('model.pkl', 'rb')
model=pkl.load(p)
p.close()
#a = copy(tweetTexts[3][0])
#shuffle(a)
#print(tweetTexts[3][0],a)
#print(docModel[tweetTexts[3]])
#print(docModel[analyzedDocument(a, [12])])
print('Model Loaded')
def tweet2vec(tweet,model):
    text = tweet['text']
    tVec = re.findall(r"[\w']+", text)
    out = []
    for word in tVec:
        if word in model:
            out.append(model[word])
        else:
            0
            #print(word)
    #print(np.shape(tVec),np.shape(out))
    if out:
        return list(np.mean(out,0))
    else:
        return 0
def tweet2docVec(tweet,docModel):
    text = tweet['text']
    words = re.findall(r"[\w']+", text)
    tweetText = analyzedDocument(words, tweet['docTag'])
    t2 = []
    for t in tweetText[0]:
        if t in docModel.wv.vocab:
            #print(t)
            t2.append(t)
    if t2:
        #print(list(np.mean(docModel[t2],0)))
        return list(np.mean(docModel[t2],0))
    else:
        return 0


c=0
used = list(range(len(tweets)))
for tweet in tweets:
    c+=1
    if not c%int(len(tweets)/500):
        print('Embed: '+str(c/len(tweets)))
    #print(tweets.index(tweet))
    a = (tweet2vec(tweet,model))
    if a != 0:
        tweet['embed_val'] = a
    else:
        tweet['used']= 0
        used.remove(tweets.index(tweet))

    #print(tweetText)
    a = (tweet2docVec(tweet,docModel))
    #print(a)
    if a:
        tweet['doc_embed']=a
    else:
        tweet['used'] = 0
        used.remove(tweets.index(tweet))
print('Embedded')
#print(len(tweets[used[9978]]['embed_val']+tweets[used[9978]]['doc_embed']))
posI = []
negI = []
c=0
#used = []
#for tweet in tweets:
#    c+=1
#    if not c%int(len(tweets)/500):
#        print('Used: '+str(c/len(tweets)))
#    if tweet['used']:
#        used.append(tweets.index(tweet))
'''for c in used:
    tweet = tweets[c]
    if not c%int(len(tweets)/500):
        print('Sentiment: '+str(c/len(tweets)))
    if tweet['sentiment']['SentimentScore']['Positive']>0.5:
        #print(tweets.index(tweet), tweet['sentiment']['SentimentScore']['Positive'],tweet['tweet']['text'])
        posI.append(tweets.index(tweet))
    if tweet['sentiment']['SentimentScore']['Negative']>0.5:
        #print(tweets.index(tweet), tweet['sentiment']['SentimentScore']['Negative'],tweet['tweet']['text'])
        negI.append(tweets.index(tweet))
print(posI)
print(negI)

p=open('toTrain.pkl', 'wb')
pkl.dump((posI,negI),p)
p.close()
'''
p=open('toTrain.pkl', 'rb')
(posI,negI)=pkl.load(p)
p.close()
print('Loaded')
#posI = posI[0:1000]
#negI = negI[0:1000]

shuffle(posI)
shuffle(negI)

even = min(len(posI),len(negI))
print(even)

classifier = svm.LinearSVC()
posEmb=[tweets[posI[i]]['embed_val']+tweets[posI[i]]['doc_embed'] for i in range(even)]
negEmb=[tweets[negI[i]]['embed_val']+tweets[negI[i]]['doc_embed'] for i in range(even)]
EmbTrain = posEmb + negEmb
posC = [1 for i in range(even)]
negC = [-1 for i in range(even)]
CTrain = posC + negC
EmbTrain = np.asmatrix(EmbTrain)

classifier.fit(EmbTrain,CTrain)
p=open('LSVM.pkl', 'wb')
pkl.dump(classifier,p)
p.close()

p=open('LSVM.pkl', 'rb')
classifier=pkl.load(p)
p.close()



print('SVM Fit')
cor = 0
tot = 0

Embeddings= [tweets[used[i]]['embed_val']+tweets[used[i]]['doc_embed'] for i in range(len(used))]
#print(sum(classifier.predict(Embeddings)))
for t in used:
    if not (t+1)%int(len(used)/10):
        print('Current Acc: ', cor/tot)
    if not t%int(len(used)/500):
        print(t/len(used))
    i = used.index(t)
    if tweets[t]['sentiment']['Sentiment'] != 'NEUTRAL' and tweets[t]['sentiment']['Sentiment'] != 'MIXED':
        tot+=1
        #print(i, Embeddings[i:i+1])
        if (tweets[t]['sentiment']['Sentiment'] == 'POSITIVE' and classifier.predict(Embeddings[i:i+1]) == [1]) or (tweets[t]['sentiment']['Sentiment'] == 'NEGATIVE' and classifier.predict(Embeddings[i:i+1]) == [-1]):
            cor+=1
print(cor/tot)
