# encoding=utf-8
import torch
from torch import nn
import argparse

cuda_condition = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda_condition else 'cpu')

parser = argparse.ArgumentParser(description='transformer xl')
parser.add_argument('--vocab_size', default=21128)
parser.add_argument('--hidden_size', default=768)
parser.add_argument('--num_layers', default=12, help='encoder layers')
parser.add_argument('--mul_attention_heads', default=8)
parser.add_argument('--intermediate_size', default=3072)
parser.add_argument('--max_position_ids', default=1000, help='max sentence long')
parser.add_argument('--type_vocab_size', default=2)
parser.add_argument('--dropout', default=0.1)
parser.add_argument('--eps', default=1e-12, help='layer norm eps')
parser.add_argument('--batch_size', default=2)
parser.add_argument('--epochs', default=1)
parser.add_argument('--learn_rate', default=1e-5)
parser.add_argument('--sentence_len', default=64)
parser.add_argument('--mem_len', default=128)
parser.add_argument('--max_position_tokens', default=1000)
parser.add_argument('--device', default=device)
parser.add_argument('--pad_idx', default=0)
parser.add_argument('--classify', default=2)
parser.add_argument('--vocab_path', default='pretrain/vocab.txt')
parser.add_argument('--train_path', default='data/train')
parser.add_argument('--dev_path', default='data/dev')
parser.add_argument('--test_path', default='data/test')
parser.add_argument('--save_path', default='finetune')
args = parser.parse_args()
















