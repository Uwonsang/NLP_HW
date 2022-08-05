import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

from collections import Counter
from torch.utils.data import WeightedRandomSampler
import os

import utils
import network



'''parameter'''
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

batch_size = 256
num_epochs = 20
learning_rate = 0.001

'''load data'''
# [0 : 952], [1 : 1031], [2 : 71], [3 : 1001], [4: 746] [5 : 599]
train_sents, train_labels = utils.load_data(filepath='../data/sent_class.train.csv')
test_sents, test_labels = utils.load_data(filepath='../data/sent_class.test.csv')

'''define sampler'''
class_sample_count = np.array([len(np.where(train_labels == t)[0]) for t in np.unique(train_labels)])
class_weight = 1. / class_sample_count
sample_weight = torch.from_numpy(np.array([class_weight[t] for t in train_labels]))
sampler = WeightedRandomSampler(torch.DoubleTensor(sample_weight), len(sample_weight))

'''tokenization'''
train_tokens = utils.tokenization(train_sents)
test_tokens = utils.tokenization(test_sents)

'''lemmatization'''
train_lemmas = utils.lemmatization(train_tokens)
test_lemmas = utils.lemmatization(test_tokens)
total_lemmas = train_lemmas + test_lemmas

'''make char_dict'''
total_dict = utils.make_dict(total_lemmas)

#character int representation
total_char_int = utils.char_int(total_lemmas, total_dict)
train_char_int = total_char_int[:len(train_lemmas)]
test_char_int = total_char_int[len(train_lemmas):]

train_char_int, train_labels = torch.LongTensor(train_char_int).to(device), torch.LongTensor(train_labels).to(device)
test_char_int = torch.LongTensor(test_char_int).to(device)

train_dataset = TensorDataset(train_char_int, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=sampler)

'''load model'''
'''train_char_int.shape = [4500 , 27(sentence_length) , 28(word_length)]'''
vocab_size = len(total_dict)
sentence_length = train_char_int.shape[1]
word_length = train_char_int.shape[2]
model = network.simple_CNN(vocab_size, batch_size, sentence_length, word_length).to(device)

'''set loss and optimizer'''
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# # sequence classification
for epoch in range(num_epochs):

    for batch_idx, (train_batch_x, train_batch_y) in enumerate(train_loader):
        predict_y = model(train_batch_x)
        loss = criterion(predict_y, train_batch_y)

        predict_index = torch.argmax(predict_y, -1)
        acc = torch.sum(predict_index == train_batch_y) / len(train_batch_y)
        acc = torch.round(acc * 100)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        print(f"Epoch: {epoch} | MSE loss: {loss} | acc: {acc}")

model.eval()
with torch.no_grad():

    predict_y = model(test_char_int)
    predict_index = np.array(torch.argmax(predict_y, -1).cpu())

    id_set = [ str(i) for i in range(1, len(predict_index)+1)]

    final_id = []
    for i in id_set:
        if len(i) == 1:
            final_id.append('S00' + i)
        elif len(i) == 2:
            final_id.append('S0' + i)
        else:
            final_id.append('S' + i)

    pred_df = pd.DataFrame({
        'id': final_id,
        'pred': predict_index
    })

    with open(os.path.join('predict.csv'), 'w') as f:
        f.write(pred_df.to_csv(index=False))