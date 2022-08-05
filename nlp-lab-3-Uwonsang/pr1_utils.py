import json
import numpy as np
import pandas as pd
from nltk import word_tokenize


def load_data(filepath):
    csv_data = pd.read_csv(filepath)
    data = csv_data['sentence']

    if 'label' in csv_data:
        targets = csv_data['label']
    else:
        targets = None

    return data, targets


def tokenization(sents):
    tokens = []
    for sent in sents:
        tokens.append(word_tokenize(sent))

    return tokens


def load_embedding(file_path):

    with open(file_path) as json_file:
        json_data = json.load(json_file)

    return json_data


def char_int(data_set, embedding_weight):

    word_to_id = {}
    for w in embedding_weight.keys():
        word_to_id[w] = len(word_to_id)

    id_to_word = {i: w for w, i in word_to_id.items()}


    ### data into int & padding
    integer_encoded = [[] for i in range(len(data_set))]

    word_max_len = 20

    for i, data in enumerate(data_set):
        row = [word_to_id[word] if word in word_to_id.keys() else word_to_id['[UNK]'] for word in data]
        ### word_pre_padding
        padding_list = [0] * (word_max_len - len(row))
        final_row = padding_list + row
        integer_encoded[i].append(final_row[:word_max_len])

    integer_encoded = np.array(integer_encoded).reshape(-1, word_max_len)

    return integer_encoded