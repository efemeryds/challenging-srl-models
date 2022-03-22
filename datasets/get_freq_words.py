"""
Getting wikipedia corpus from https://dumps.wikimedia.org/enwiki/latest/
Getting frequencies by using this library https://github.com/IlyaSemenov/wikipedia-word-frequency
"""

import json
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

# download these packages before the first run
# nltk.download('brown')
# nltk.download('universal_tagset')


# load the list of words
data = pd.read_csv('wordfreq.txt', header=None)
data = data[0].str.rsplit(' ', 1, expand=True)
data.columns = ['words', 'freq']
data['freq'] = data['freq'].astype(int)

# filter the words -> leave bottom starting from frequency 1000
tmp_data = data[data['freq'] >= 5000]
tmp_data = tmp_data[tmp_data['freq'] <= 100000]

# plot the frequencies and remove outliers
sns.distplot(tmp_data['freq'])
# plt.show()

sns.boxplot(tmp_data['freq'])
# plt.show()

# get the brown corpus to tag words relatively to it
wordtags = nltk.ConditionalFreqDist((w.lower(), t)
                                    for w, t in nltk.corpus.brown.tagged_words(tagset="universal"))

tmp_tag_list = []
for k in range(len(tmp_data)):
    word = tmp_data['words'].iloc[k]
    # take first tag
    if len(list(wordtags[word])) > 0:
        tag = list(wordtags[word])[0]
    else:
        tag = ''
    tmp_tag_list.append(tag)

tmp_data['tags'] = tmp_tag_list

# save proper lists into new files

top_freq_nouns = list(tmp_data[tmp_data['tags'] == 'NOUN'].head(100)['words'])
bottom_freq_nouns = list(tmp_data[tmp_data['tags'] == 'NOUN'].tail(100)['words'])

top_freq_verbs = list(tmp_data[tmp_data['tags'] == 'VERB'].head(100)['words'])
bottom_freq_verbs = list(tmp_data[tmp_data['tags'] == 'VERB'].tail(100)['words'])

data = {
    'data': [
        {
            'top_freq_nouns': top_freq_nouns,
            'bottom_freq_nouns': bottom_freq_nouns,
            'top_freq_verbs': top_freq_verbs,
            'bottom_freq_verbs': bottom_freq_verbs

        }
    ]
}
with open('final_vocab.json', 'w', encoding ='utf8') as json_file:
    json.dump(data, json_file, allow_nan=True)

print('DONE')
