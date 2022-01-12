# encoding=utf-8
from token_model.tokenization import Tokenizer
from config import *
import torch
import pandas as pd


class BuildData(object):
    def __init__(self):
        self.cls = ['[CLS]']
        self.sep = ['[SEP]']
        self.Tokenizer = Tokenizer(args.vocab_path)
        self.sentence_len = args.sentence_len

    def convert_data(self, text):
        tokens = self.Tokenizer.tokenize(text)
        tokens = self.cls + tokens + self.sep
        text_ids = self.Tokenizer.convert_tokens_to_ids(tokens)
        if len(tokens) < args.sentence_len:
            text_ids += (args.sentence_len - len(tokens)) * [0]
        else:
            text_ids = text_ids[:args.sentence_len]
        return text_ids

    def load_data(self, path):
        data = pd.read_csv(path)
        output = []
        for i, row in data.iterrows():
            text_a, text_b, label = row['question'], row['syn_question'], row['label']
            text_aids = self.convert_data(text_a)
            text_bids = self.convert_data(text_b)
            output.append((text_aids, text_bids, int(str(label).strip())))
        return output

    def build_data(self):
        train_data = self.load_data('data/train.csv')
        dev_data = self.load_data('data/dev.csv')
        return train_data, dev_data


class BatchData(object):
    def __init__(self, data, index=0, batch_size=args.batch_size):
        self.index = index
        self.device = args.device
        self.batch_size = batch_size
        self.data = data
        self.batch_nums = len(self.data) // self.batch_size
        self.residue = False
        if len(self.data) % self.batch_size != 0:
            self.residue = True

    def to_tensor(self, batch):
        text_a = torch.LongTensor([_[0] for _ in batch]).to(self.device)
        text_b = torch.LongTensor([_[1] for _ in batch]).to(self.device)
        label = torch.LongTensor([_[2] for _ in batch]).to(self.device)
        return text_a, text_b, label

    def __next__(self):
        if self.residue and self.index == self.batch_nums:
            batch = self.data[self.index*self.batch_size:len(self.data)]
            self.index += 1
            return self.to_tensor(batch)
        elif self.index >= self.batch_nums:
            self.index = 0
            raise StopIteration
        else:
            batch = self.data[self.index*self.batch_size:(self.index+1)*self.batch_size]
            self.index += 1
            return self.to_tensor(batch)

    def __iter__(self):
        return self


if __name__ == '__main__':
    # path = args.train_path
    pp = BuildData()
    train, _ = pp.build_data()
    # print(train)
    train_pp = BatchData(train)
    for k in train_pp:
        print('k', k[0])
        print('b', k[1])
        print('v', k[2])
    #







