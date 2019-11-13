import numpy as np
import os.path
import urllib.request

url = 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/'
file_names = [
    'ptb.train.txt',
    'ptb.test.txt',
    'ptb.valid.txt'
]

def download(url, file_from, file_to):
    if not os.path.exists(file_to):
        #import ssl
        #ssl._create_default_https_context = ssl._create_unverified_context
        urllib.request.urlretrieve(url + file_from, file_to)
    return file_to

def get_vocab(url, file_name):
    local = download(url, file_name, './' + file_name)
    words = open(local).read().replace('\n', '<eos>').strip().split()
    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            wid = len(word_to_id)
            word_to_id[word] = wid
            id_to_word[wid] = word
    corpus = np.array([word_to_id[w] for w in words])
    return (corpus, word_to_id, id_to_word)
