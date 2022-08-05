import data_load
import matplotlib.pyplot as plt
from model import *
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import os
import pandas as pd
import numpy as np
from torch.utils.data import WeightedRandomSampler


os.environ["CUDA_VISIBLE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

'''parameter'''
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

batch_size = 20
num_epochs = 100
learning_rate = 0.01
decay = 0.99


'''sybols of symbol sequence into One-hot representation & vectorize symbol sequence'''
class preprocessing:

    def __init__(self, train_data, test_data):

        self.train_symbol_list = sorted(set(sum(train_data, [])))
        self.test_symbol_list = sorted(set(sum(test_data, [])))

        self.train_word_to_id = {"PAD": 0}
        for w in self.train_symbol_list:
            self.train_word_to_id[w] = len(self.train_word_to_id)
        self.train_id_to_word = {i: w for w, i in self.train_word_to_id.items()}


    def make_train_dict(self):
        return self.train_word_to_id

    def make_total_dict(self):
        add_vocab = set(self.test_symbol_list) - set(self.train_symbol_list)
        add_vocab_char_to_int = dict((c, i) for i, c in enumerate(add_vocab, start=len(self.train_word_to_id)))

        self.test_word_to_id = dict(self.train_word_to_id, **add_vocab_char_to_int)
        self.test_id_to_word = {i: w for w, i in self.test_word_to_id.items()}

        return self.test_word_to_id


def one_hot_representation(raw_data, dictionary):

    ### data into int & padding
    max_len = max(len(i) for i in raw_data)
    integer_encoded = [[] for i in range(len(raw_data))]
    for i, data in enumerate(raw_data):
        row = [dictionary[char] for char in data]
        ### padding
        row += [0] * (max_len - len(row))
        integer_encoded[i].append(row)

    ### vectorize symbol sequence
    onehot_metrix = np.eye(len(dictionary))
    onehot_encoded = onehot_metrix[integer_encoded]
    onehot_encoded = np.array(onehot_encoded).reshape(-1, max_len, len(dictionary))

    return onehot_encoded

def plot(loss):
    plt.plot(loss)
    plt.title('loss')
    plt.xlabel('epoch')
    plt.show()

def main(train_file, test_file):

    '''load data'''
    train_x, train_y = data_load.symbol_dataset(train_file, mode='train').getdataset()
    test_x, _ = data_load.symbol_dataset(test_file, mode='test').getdataset()

    '''define sampler'''
    class_sample_count = np.array([len(np.where(train_y == t)[0]) for t in np.unique(train_y)])
    class_weight = 1./ class_sample_count
    sample_weight = torch.from_numpy(np.array([class_weight[t] for t in train_y]))
    sampler = WeightedRandomSampler(torch.DoubleTensor(sample_weight), len(sample_weight))

    preprocessing_data = preprocessing(train_x, test_x)
    total_dict = preprocessing_data.make_total_dict()

    train_onehot_x = one_hot_representation(train_x, total_dict).reshape(len(train_x), -1)
    test_onehot_x = one_hot_representation(test_x, total_dict).reshape(len(test_x), -1)

    train_onehot_x, train_y = torch.FloatTensor(train_onehot_x).to(device), torch.LongTensor(train_y).to(device)
    test_onehot_x = torch.FloatTensor(test_onehot_x).to(device)

    train_dataset = TensorDataset(train_onehot_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=sampler)

    '''load model'''
    vocab_size = len(total_dict)
    max_length = 20
    model = simple_MLP(vocab_size * max_length).to(device)

    '''set loss and optimizer'''
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


    for epoch in range(num_epochs):

        for batch_idx, (train_batch_x, train_batch_y) in enumerate(train_loader):
            predict_y = model(train_batch_x)
            loss = criterion(predict_y, train_batch_y)

            predict_index = torch.argmax(predict_y, -1)
            acc = torch.sum(predict_index == train_batch_y) / len(train_batch_y)
            acc = torch.round(acc*100)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            print(f"Epoch: {epoch} | MSE loss: {loss} | acc: {acc}")


    model.eval()
    with torch.no_grad():

        predict_y = model(test_onehot_x)
        predict_index = np.array(torch.argmax(predict_y, -1).cpu())
        target_dict = data_load.symbol_dataset(train_file, mode='train').target_dict()
        target = [target_dict[data] for data in predict_index]

        id_set = [ str(i) for i in range(1, len(target)+1)]

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
            'pred': target
        })

        with open(os.path.join('predict.csv'), 'w') as f:
            f.write(pred_df.to_csv(index=False))


if __name__ == '__main__':
    train_name = './dataset/simple_seq.train.csv'
    test_name= './dataset/simple_seq.test.csv'
    main(train_name, test_name)

