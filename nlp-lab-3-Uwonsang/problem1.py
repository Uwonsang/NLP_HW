import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

import os

import pr1_utils
import network

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

'''parameter'''
batch_size = 256
num_epochs = 200
learning_rate = 0.00001

'''load data'''
train_sentence, train_labels = pr1_utils.load_data(filepath='./dataset/classification/classification_datasets/train_set.csv')
test_sentence, test_labels = pr1_utils.load_data(filepath='./dataset/classification/classification_datasets/test_set.csv')

'''1.1 tokenization'''
train_tokens = pr1_utils.tokenization(train_sentence)
test_tokens = pr1_utils.tokenization(test_sentence)

'''1.3 use given glove word embedding dict & word'''
embedding_weight = pr1_utils.load_embedding(file_path='./dataset/classification/classification_datasets/glove_word.json')
embedding_vector = pd.DataFrame.from_dict(embedding_weight).T.reset_index(drop=True)
embedding_vector = torch.from_numpy(embedding_vector.to_numpy()).float()

'''1.2 data_preprocessing(pre_padding & pre-sequence truncation)'''
train_data = pr1_utils.char_int(train_tokens, embedding_weight)
test_data = pr1_utils.char_int(test_tokens, embedding_weight)

train_data, train_labels = torch.LongTensor(train_data).to(device), torch.LongTensor(train_labels).to(device)
test_data = torch.LongTensor(test_data).to(device)

train_dataset = TensorDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = network.simple_RNN(embed_weight_vec=embedding_vector, device=device).to(device)

'''set loss and optimizer'''
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

'''2.Train classification model'''
for epoch in range(num_epochs):

    for batch_idx, (train_batch_x, train_batch_y) in enumerate(train_loader):

        predict_y = model(train_batch_x)
        loss = criterion(predict_y, train_batch_y)

        predict_index = torch.argmax(predict_y, -1)
        acc = torch.sum(predict_index == train_batch_y) / len(train_batch_y)
        acc = torch.round(acc * 100)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        print(f"Epoch: {epoch} | MSE loss: {loss} | acc: {acc}")


model.eval()
with torch.no_grad():

    predict_y = model(test_data)
    predict_index = np.array(torch.argmax(predict_y, -1).cpu())

    id_set = [ str(i) for i in range(0, len(predict_index))]

    final_id = []
    for i in id_set:
        final_id.append('S' + i)

    pred_df = pd.DataFrame({
        'ID': final_id,
        'label': predict_index
    })

    with open(os.path.join('predict.csv'), 'w') as f:
        f.write(pred_df.to_csv(index=False))