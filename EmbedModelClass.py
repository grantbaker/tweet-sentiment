import pandas as pd
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

class GrantMattModel():
    def __init__(self, svm='LSVM'):
        with open('docModel.pkl','rb') as f:
            self.docModel = pkl.load(f)
        print('Doc Model Loaded')
        with open('model.pkl','rb') as f:
            self.model = pkl.load(f)
        print('Model Loaded')
        if svm == 'SVM':
            with open('SVM.pkl', 'rb') as f:
                self.sentiment = pkl.load(f) 
        else:
            with open('LSVM.pkl', 'rb') as f:
                self.sentiment = pkl.load(f)
    '''def train(self, features, embeds, labels):
        cores = 1
        self.vectorizer = DictVectorizer(sparse=False)
        self.sentiment = BaggingClassifier(svm.LinearSVC(), max_samples=1.0/cores,
                                           n_estimators=cores, n_jobs=cores)

        feat_vec = self.vectorizer.fit_transform(features)

        embeds = np.concatenate((embeds, feat_vec), axis=1)
        self.sentiment.fit(embeds, labels)
        '''

    def extract_features(self, tweet):

        text = tweet
        tVec = re.findall(r"[\w']+", text)
        out = []
        for word in tVec:
            if word in self.model:
                out.append(self.model[word])
            else:
                0
        if out:
            t1 = list(np.mean(out,0))
        else:
            t1 = list(np.zeros(300))

        words = re.findall(r"[\w']+", text)
        t2 = []
        for t in words:
            if t in self.docModel.wv.vocab:
                #print(t)
                t2.append(t)
        if t2:
            #print(list(np.mean(docModel[t2],0)))
            t2 = list(np.mean(self.docModel[t2],0))
        else:
            t2 = list(np.zeros(100))
        feats = t1+t2
        return feats

    def predict(self, newTweet):
        feats = self.extract_features(newTweet)
        #print(feats)
        return self.sentiment.decision_function(np.array(feats).reshape(1, -1))[0]

    def evaluate(self, test_df):
        test_labels = test_df.as_matrix(columns=['sentiment']).flatten()/2-1
        test_embeds = np.zeros((len(test_df), 400))
        for i in range(len(test_df)):
            test_embeds[i] = self.extract_features(test_df['text'][i])
        test_vec = test_embeds

        acc = self.sentiment.score(test_vec, test_labels)
#         pred_labels = np.tanh(self.sentiment.decision_function(feat_vec))
#         conf_acc = (test_labels + pred_labels)/(2*len(test_df))
#         return (acc, conf_acc)
        return acc
