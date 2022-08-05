import json
import numpy as np
import pandas as pd
from nltk import word_tokenize


def load_data(file_path):

    with open(file_path) as json_file:
        json_data = json.load(json_file)

    pd_data = pd.DataFrame.from_dict(json_data).T
    data = pd_data['tokens']

    if 'ud_tags' in pd_data:
        targets = pd_data['ud_tags']
    else:
        targets = None

    return data, targets


def label2int(label_dataset, label_dict_path, mode):

    label_pd = pd.read_csv(label_dict_path, header=None)
    label_dict_num = label_pd[0].to_dict()
    label_dict_char = { value : key for key,value in label_dict_num.items()}

    if mode == 'train':
        word_max_len = 20
    elif mode == 'test':
        word_max_len = max(len(sentence) for sentence in label_dataset)

    label_int = []
    for data in label_dataset:
        row = np.array([label_dict_char[pos] for pos in data])
        row = np.append(row, [0] * (word_max_len - len(row)))
        label_int.append(np.array(row[:word_max_len]))

    label_int = np.array(label_int).reshape(-1, word_max_len)

    return label_int


def load_embedding(file_path):

    with open(file_path) as json_file:
        json_data = json.load(json_file)

    return json_data


def char_int(data_set, embedding_weight, mode):

    word_to_id = {}
    for w in embedding_weight.keys():
        word_to_id[w] = len(word_to_id)

    id_to_word = {i: w for w, i in word_to_id.items()}

    ### data into int & padding
    integer_encoded = [[] for i in range(len(data_set))]

    if mode == 'train':
        word_max_len = 20
    elif mode == 'test':
        word_max_len = max(len(word) for word in sum(data_set, []))

    for i, data in enumerate(data_set):
        row = [word_to_id[word] if word in word_to_id.keys() else word_to_id['[UNK]'] for word in data]
        ### word_post_padding
        final_row = row + [0] * (word_max_len - len(row))
        integer_encoded[i].append(final_row[:word_max_len])

    integer_encoded = np.array(integer_encoded).reshape(-1, word_max_len)

    return integer_encoded


def masking(data_set):
    masked = data_set.gt(0)
    return masked