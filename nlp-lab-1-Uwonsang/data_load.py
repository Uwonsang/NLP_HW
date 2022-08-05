import csv
import numpy as np


class symbol_dataset:

    def __init__(self, filename, mode):

        '''1.Load input symbol sequence'''
        with open(filename, encoding="utf8") as f:
            reader = csv.reader(f)
            csv_list = []
            for i in reader:
                csv_list.append(i)
            f.close()

        filter_symbol = [[] for i in range(len(csv_list))]
        for i, dataset in enumerate(csv_list):
            for symbol in dataset:
                # if (symbol != None) and (symbol != ''):
                if symbol != '':
                    filter_symbol[i].append(symbol)

        X = []
        Y = []
        for data in filter_symbol:
            if mode == 'train':
                X.append(data[:-1])
                Y.append(data[-1])
            else:
                X.append(data[:])

        self.y_dict = dict((c, i) for i, c in enumerate(sorted(set(Y))))

        self.X = X
        self.Y = np.array([self.y_dict[data] for data in Y])

    def getdataset(self):
        return self.X, self.Y

    def target_dict(self):
        return dict((i, c) for i, c in enumerate(self.y_dict))


if __name__ == '__main__':
    dataset = symbol_dataset('./dataset/simple_seq.train.csv')
