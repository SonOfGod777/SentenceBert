# encoding=utf-8
import torch
import sys
import numpy as np
from sklearn import metrics
from torch import nn
from torch.optim import Adam
from pre_data import BuildData, BatchData
from config import args
from bert_model.bert_model import BertModel


class Train(object):
    def __init__(self):
        train_data, dev_data = BuildData().build_data()
        self.train_data = BatchData(train_data)
        self.dev_data = BatchData(dev_data)
        self.BertModel = BertModel()
        self.optimizer = Adam(self.BertModel.parameters(), lr=args.learn_rate)
        self.loss = nn.CrossEntropyLoss().to(args.device)

    def train(self):
        total_batch = 0
        last_improve = 0  # 上次loss下降的batch数
        flag = False  # 如果loss很久没有下降，结束训练
        best_loss = float('inf')
        self.BertModel.train()
        for epoch in range(args.epochs):
            if flag:
                break
            for i, (text_a, text_b, labels) in enumerate(self.train_data):
                outputs = self.BertModel(text_a, text_b)
                loss = self.loss(outputs, labels)
                self.BertModel.zero_grad()
                loss.backward()
                self.optimizer.step()
                if total_batch % 10 == 0:
                    true_label = labels.data.cpu()
                    predict_label = torch.max(outputs.data, 1)[1].cpu()   # 0是最大值，1是最大值索引
                    train_acc = metrics.accuracy_score(true_label, predict_label)
                    # print('true_label', labels, labels.size())
                    # print('output', outputs, outputs.size())
                    # print('predict_label', predict_label, true_label)
                    dev_acc, dev_loss = self.evaluate(self.BertModel, self.dev_data)
                    if dev_loss < best_loss:
                        best_loss = dev_loss
                        save_path = args.save_path + '/trans_point.ep{}'.format(total_batch)
                        torch.save(self.BertModel.state_dict(), save_path)
                        last_improve = total_batch
                    print('epoch:{}, train_loss:{}, train_acc:{}, dev_loss:{}, dev_acc:{}, last_improve:{}'.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, total_batch))
                total_batch += 1
                if total_batch - last_improve > 1000:
                    flag = True
                    break

    def test(self, model, data):
        model.load_state_dict(torch.load(args.save_path))
        model.eval()
        test_acc, test_loss = self.evaluate(model, self.test_data)

    def evaluate(self, model, data):
        model.eval()
        loss_total = 0
        predict_all = np.array([], dtype=int)
        label_all = np.array([], dtype=int)
        with torch.no_grad():
            for i, (text_a, text_b, labels) in enumerate(data):
                output = model(text_a, text_b)
                predict_label = torch.max(output, 1)[1].cpu().numpy()
                loss = nn.CrossEntropyLoss()(output, labels).to(args.device)
                loss_total += loss
                labels = labels.cpu().numpy()
                predict_all = np.append(predict_all, predict_label)
                label_all = np.append(label_all, labels)
        acc = metrics.accuracy_score(label_all, predict_all)
        return acc, loss_total/i


if __name__ == '__main__':
    Train().train()



