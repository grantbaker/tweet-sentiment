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
from EmbedModelClass import GrantMattModel

cols = ['sentiment','id','date','query_string','user','text']
tweet_df = pd.read_csv('training.1600000.processed.noemoticon.csv',
                              header=None, names=cols, encoding='utf-8')
tweet_df.drop(['id','date','query_string','user'], axis=1, inplace=True)
labels = tweet_df.as_matrix(columns=['sentiment']).flatten()/2-1
print('Tweets Loaded')
#print(tweet_df[:1000])

gm = GrantMattModel(svm='LSVM')
for i in range(1):
    #new_df = tweet_df.sample(1000)
    #new_df = new_df.reset_index(drop=True)
    #print(new_df)
    print(gm.evaluate(tweet_df))
