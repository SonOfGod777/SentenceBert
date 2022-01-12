# encoding=utf-8
import torch
from torch import nn
import os, sys
sys.path.append('bert_model')
from embedding import TokenEmbedding
from config import *
from bert_layer import BertLayer
from LayerNorm import clones, Classify, BertPooling


class BertModel(nn.Module):
    def __init__(self):
        super(BertModel, self).__init__()
        self.all_layers = []
        self.token_embedding = TokenEmbedding()
        self.bert_layer = BertLayer()
        self.bert_layers = clones(self.bert_layer, args.num_layers)
        self.classify = Classify(args.hidden_size, args.classify)

    def get_src_mask(self, seq, pad_idx):
        src_mask = (seq != pad_idx).unsqueeze(1)
        return src_mask.int()

    def get_trg_mask(self, trg, pad_idx):
        batch, trg_len = trg.size()
        trg_mask = (trg != pad_idx).unsqueeze(-2)
        trg_mask = trg_mask & (1 - torch.triu(torch.ones(1, trg_len, trg_len), diagonal=1))
        return trg_mask

    def mean_pooling(self, output_embedding, attention_mask):
        attention_mask = attention_mask.transpose(1, 2)
        mul_mask = lambda x, m: x * m
        reduce_mask = lambda x, m: torch.sum(mul_mask(x, m), dim=1) / torch.sum(m, dim=1)
        return reduce_mask(output_embedding, attention_mask)

    def bert_model(self, input_ids, attention_mask=None, return_all_layers=True):
        input_embedding = self.token_embedding(input_ids)
        if attention_mask is None:
            attention_mask = self.get_src_mask(input_ids, args.pad_idx)

        for layer in self.bert_layers:
            input_embedding = layer(input_embedding, attention_mask)

            if return_all_layers:
                self.all_layers.append(input_embedding)
        if not return_all_layers:
            self.all_layers.append(input_embedding)
        return self.all_layers, attention_mask

    def forward(self, input_ids1, input_ids2):
        output1, attention_mask1 = self.bert_model(input_ids1)
        output2, attention_mask2 = self.bert_model(input_ids2)
        # cls
        # output1_embedding = output1[-1][:, 0]
        # output2_embedding = output2[-1][:, 0]
        # mean
        print(output1[-1].size())
        output1_embedding = self.mean_pooling(output1[-1], attention_mask1)
        output2_embedding = self.mean_pooling(output2[-1], attention_mask2)
        print(output2_embedding.size())
        output_embedding = torch.cat([output1_embedding, output2_embedding,
                                      torch.multiply(output1_embedding, output2_embedding),
                                      torch.abs(torch.subtract(output1_embedding, output2_embedding))], dim=-1)
        last_dim = output_embedding.size()[-1]
        output_embedding = BertPooling(last_dim, args.hidden_size)(output_embedding)
        return self.classify(output_embedding)


# if __name__ == '__main__':
#     input_ids = torch.arange(10).view(2, 5)
#     pp, _ = BertModel()(input_ids)
#     print(pp.size(), pp.type())












