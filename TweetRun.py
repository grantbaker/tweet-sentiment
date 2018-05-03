import pandas as pd

from EmbedModelClass import GrantMattModel
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

cols = ['sentiment','id','date','query_string','user','text']
tweet_df = pd.read_csv('training.1600000.processed.noemoticon.csv',
                              header=None, names=cols, encoding='utf-8')
tweet_df.drop(['id','date','query_string','user'], axis=1, inplace=True)
labels = tweet_df.as_matrix(columns=['sentiment']).flatten()/2-1
print('Tweets Loaded')
#print(tweet_df[:1000])

gm = GrantMattModel(svm='LSVM')




objects = list(range(300))
y_pos = np.arange(len(objects))

words = ['happy','glad','angry','upset']
print(gm.predict(''))
for word in words:
    performance = gm.extract_features_300(word)

    plt.bar(y_pos, performance, align='center', alpha=0.5)

    plt.title(word)

    plt.show()
    for word2 in words:
        if word != word2:
            perf2 = gm.extract_features_300(word2)
            print(word,word2,cosine_similarity(np.matrix(performance).reshape(1, -1),np.matrix(perf2).reshape(1, -1)))
