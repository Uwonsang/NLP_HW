import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

import os
import json

import pr2_utils
import network

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

'''parameter'''
batch_size = 256
num_epochs = 50
learning_rate = 0.0001

'''1. load data (tokenization data)'''
train_sentence, train_labels = pr2_utils.load_data(file_path='./dataset/pos/pos_datasets/train_set.json')
test_sentence, test_labels = pr2_utils.load_data(file_path='./dataset/pos/pos_datasets/test_set.json')

'''label data'''
train_labels = pr2_utils.label2int(train_labels, label_dict_path='./dataset/pos/pos_datasets/tgt.txt', mode='train')


'''3. Use Fasttext word embedding dict and vectors'''
embedding_weight = pr2_utils.load_embedding(file_path='./dataset/pos/pos_datasets/fasttext_word.json')
embedding_vector = pd.DataFrame.from_dict(embedding_weight).T.reset_index(drop=True)
embedding_vector = torch.from_numpy(embedding_vector.to_numpy()).float()


'''2. data_preprocessing(post-padding & pre-sequence truncation)'''
train_data = pr2_utils.char_int(train_sentence, embedding_weight, mode='train')
test_data = pr2_utils.char_int(test_sentence, embedding_weight, mode='test')

train_data, train_labels = torch.LongTensor(train_data), torch.LongTensor(train_labels).to(device)
test_data = torch.LongTensor(test_data)

train_dataset = TensorDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = network.Bidirectional_RNN(embed_weight_vec=embedding_vector, device=device).to(device)

'''set loss and optimizer'''
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

''' 6. Training POS Tagging model'''
for epoch in range(num_epochs):

    for batch_idx, (train_batch_x, train_batch_y) in enumerate(train_loader):

        '''5. masking'''
        masking = pr2_utils.masking(train_batch_x).view(-1)
        predict_y = model(train_batch_x)

        ##flatten##
        predict_y = predict_y.reshape(-1, predict_y.shape[-1])
        train_batch_y = train_batch_y.view(-1)

        loss = criterion(predict_y[masking], train_batch_y[masking])

        predict_index = torch.argmax(predict_y[masking], -1)
        acc = torch.sum(predict_index == train_batch_y[masking]) / len(train_batch_y[masking])
        acc = torch.round(acc * 100)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        print(f"Epoch: {epoch} | MSE loss: {loss} | acc: {acc}")


model.eval()
with torch.no_grad():

    test_masking = pr2_utils.masking(test_data).view(-1)

    predict_y = model(test_data)
    predict_y = predict_y.reshape(-1, predict_y.shape[-1])


    predict_index = np.array(torch.argmax(predict_y[test_masking], -1).cpu())

    id_set = [ str(i) for i in range(0, len(predict_index))]

    final_id = []
    for i in id_set:
        final_id.append('S' + i)

    pred_df = pd.DataFrame({
        'ID': final_id,
        'label': predict_index
    })

    with open(os.path.join('predict_pr2.csv'), 'w') as f:
        f.write(pred_df.to_csv(index=False))