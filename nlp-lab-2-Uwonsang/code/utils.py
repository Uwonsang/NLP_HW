import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag



lemma = WordNetLemmatizer()


def load_data(filepath):
	csv_data = pd.read_csv(filepath)
	data = csv_data['sentence']
	targets = csv_data['label']

	return data, targets


def tokenization(sents):
	tokens = []
	for sent in sents:
		tokens.append(word_tokenize(sent))

	return tokens


def lemmatization(tokens):

	word2pos_set = [pos_tagging(word) for word in tokens]

	lemmas_list = []
	for word2pos in word2pos_set:
		lemmas = []
		for word, pos in word2pos.items():
			lemmas.append(lemma.lemmatize(word, pos))
		lemmas_list.append(lemmas)

	return lemmas_list

def pos_tagging(words):

	words_only_alpha = [w for w in words if w.isalpha()]

	res_pos_list = pos_tag(words_only_alpha)
	word2pos = {w: p for w, p in list(map(format_conversion, res_pos_list))}

	for w in words_only_alpha:
		if w not in word2pos:
			word2pos[w] = 'n'

	return word2pos

def format_conversion(words):

	pos_tags = ['n', 'v', 'a', 'r', 's']
	w, p = words
	p_lower = p[0].lower()
	p_new = 'n' if p_lower not in pos_tags else p_lower

	return w, p_new


######################## we need to char embedding ###################################

def make_dict(total_data):
	symbol_list = sorted(set(sum(total_data, [])))
	char_list = set([char for word in symbol_list for char in word])
	word_to_id = {"PAD": 0}

	for w in char_list:
		word_to_id[w] = len(word_to_id)
	id_to_word = {i: w for w, i in word_to_id.items()}

	return word_to_id


def char_int(data_set, dictionary):

	### data into int & padding
	integer_encoded = [[] for i in range(len(data_set))]
	
	word_max_len = max(len(word) for word in sum(data_set, []))

	for i, data in enumerate(data_set):
		word_int = []
		for word in data:
			row = [dictionary[char] for char in word]
			### char_padding
			row += [0] * (word_max_len - len(row))
			word_int.append(np.array(row))
		integer_encoded[i].append(np.array(word_int))

	integer_encoded = np.array(integer_encoded).reshape(-1)

	### sentence_padding
	sen_max_len = max(len(sentence) for sentence in integer_encoded)
	final_encoded = np.zeros((len(data_set), sen_max_len, word_max_len))

	for i, sentence in enumerate(integer_encoded):
		word_size = len(sentence[0])
		sentence_pad = np.zeros(( sen_max_len - len(sentence) , word_size), dtype=int)
		# print('sentence_pad :', sentence_pad.shape)
		sentence_with_pad = np.concatenate((sentence, sentence_pad), axis=0)
		# print('sentence_with_pad :', sentence_with_pad.shape)
		final_encoded[i] = sentence_with_pad

	### slicing word_size, sentence_size
	final_encoded = final_encoded[:,:20,:10]

	return final_encoded