import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import numpy as np
import pandas as pd
import os
import pickle
import random
import itertools
import matplotlib.pyplot as plt

prefix = os.path.abspath(os.path.join(os.getcwd(), "."))
text_features = np.load(os.path.join(prefix, 'Features/TextWhole/whole_samples_reg_avg.npz'))['arr_0']
text_targets = np.load(os.path.join(prefix, 'Features/TextWhole/whole_labels_reg_avg.npz'))['arr_0']
text_targets = np.where(text_targets >= 53, 1, 0)
text_dep_idxs_tmp = np.where(text_targets == 1)[0]
text_non_idxs = np.where(text_targets == 0)[0]

'''text_features = np.load(os.path.join(prefix, 'Features/TextWhole/whole_samples_300_word2.npz'))['arr_0'].astype('float32')
text_targets = np.load(os.path.join(prefix, 'Features/TextWhole/whole_labels_300_word2.npz'))['arr_0'].astype('int32')
text_dep_idxs_tmp = np.where(text_targets == 1)[0]
text_non_idxs = np.where(text_targets == 0)[0]'''

'''prefix = os.path.abspath(os.path.join("E:/"))
text_features = np.load(os.path.join(prefix, 'DAICCode/Features/DAIC_tesamples_clf.npz'))['arr_0']
text_targets = np.load(os.path.join(prefix, 'DAICCode/Features/DAIC_telabels_clf.npz'))['arr_0']
text_dep_idxs_tmp = np.where(text_targets == 1)[0]
text_non_idxs = np.where(text_targets == 0)[0]'''


class TextBiLSTM(nn.Module):
    def __init__(self, config):
        super(TextBiLSTM, self).__init__()
        self.num_classes = config['num_classes']
        self.learning_rate = config['learning_rate']
        self.dropout = config['dropout']
        self.hidden_dims = config['hidden_dims']
        self.rnn_layers = config['rnn_layers']
        self.embedding_size = config['embedding_size']
        self.bidirectional = config['bidirectional']

        self.build_model()
        self.init_weight()
        
    def init_weight(net):
        for name, param in net.named_parameters():
            if 'ln' not in name:
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_uniform_(param)

    def build_model(self):
        # attention layer
        self.attention_layer = nn.Sequential(
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(inplace=True)
        )
        # self.attention_weights = self.attention_weights.view(self.hidden_dims, 1)

        # 双层lstm
        self.lstm_net = nn.LSTM(self.embedding_size, self.hidden_dims,
                                num_layers=self.rnn_layers, dropout=self.dropout,
                                bidirectional=self.bidirectional)
                
        # FC层
        # self.fc_out = nn.Linear(self.hidden_dims, self.num_classes)
        self.fc_out = nn.Sequential(
            # nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dims, self.num_classes),
            # nn.ReLU(),
            nn.Softmax(dim=1),
        )

        self.ln1 = nn.LayerNorm(self.embedding_size)
        self.ln2 = nn.LayerNorm(self.hidden_dims)


    def attention_net_with_w(self, lstm_out, lstm_hidden):
        '''
        :param lstm_out:    [batch_size, len_seq, n_hidden * 2]
        :param lstm_hidden: [batch_size, num_layers * num_directions, n_hidden]
        :return: [batch_size, n_hidden]
        '''
        lstm_tmp_out = torch.chunk(lstm_out, 2, -1)
        # h [batch_size, time_step, hidden_dims]
        h = lstm_tmp_out[0] + lstm_tmp_out[1]
        # h = lstm_out
        # [batch_size, num_layers * num_directions, n_hidden]
        lstm_hidden = torch.sum(lstm_hidden, dim=1)
        # [batch_size, 1, n_hidden]
        lstm_hidden = lstm_hidden.unsqueeze(1)
        # atten_w [batch_size, 1, hidden_dims]
        atten_w = self.attention_layer(lstm_hidden)
        # m [batch_size, time_step, hidden_dims]
        m = nn.Tanh()(h)
        # atten_context [batch_size, 1, time_step]
        atten_context = torch.bmm(atten_w, m.transpose(1, 2))
        # softmax_w [batch_size, 1, time_step]
        softmax_w = F.softmax(atten_context, dim=-1)
        # context [batch_size, 1, hidden_dims]
        context = torch.bmm(softmax_w, h)
        result = context.squeeze(1)
        return result

    def forward(self, x):
        # x : [len_seq, batch_size, embedding_dim]
        x = x.permute(1, 0, 2)
        # x = self.ln1(x)
        output, (final_hidden_state, _) = self.lstm_net(x)
        # output : [batch_size, len_seq, n_hidden * 2]
        output = output.permute(1, 0, 2)
        # final_hidden_state : [batch_size, num_layers * num_directions, n_hidden]
        final_hidden_state = final_hidden_state.permute(1, 0, 2)
        # final_hidden_state = torch.mean(final_hidden_state, dim=0, keepdim=True)
        # atten_out = self.attention_net(output, final_hidden_state)
        atten_out = self.attention_net_with_w(output, final_hidden_state)
        # atten_out = self.ln2(atten_out)
        return self.fc_out(atten_out)

def save(model, filename):
    save_filename = '{}.pt'.format(filename)
    torch.save(model.state_dict(), save_filename)
    print('Saved as %s' % save_filename)
    
def standard_confusion_matrix(y_test, y_test_pred):
    """
    Make confusion matrix with format:
                  -----------
                  | TP | FP |
                  -----------
                  | FN | TN |
                  -----------
    Parameters
    ----------
    y_true : ndarray - 1D
    y_pred : ndarray - 1D

    Returns
    -------
    ndarray - 2D
    """
    [[tn, fp], [fn, tp]] = confusion_matrix(y_test, y_test_pred)
    return np.array([[tp, fp], [fn, tn]])

def model_performance(y_test, y_test_pred_proba):
    """
    Evaluation metrics for network performance.
    """
    y_test_pred = y_test_pred_proba.data.max(1, keepdim=True)[1]

    # Computing confusion matrix for test dataset
    conf_matrix = standard_confusion_matrix(y_test, y_test_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    return y_test_pred, conf_matrix

def train(epoch, train_idxs):
    global lr, train_acc
    model.train()
    batch_idx = 1
    total_loss = 0
    correct = 0
    X_train = text_features[train_idxs]
    Y_train = text_targets[train_idxs]
    for i in range(0, X_train.shape[0], config['batch_size']):
        if i + config['batch_size'] > X_train.shape[0]:
            x, y = X_train[i:], Y_train[i:]
        else:
            x, y = X_train[i:(i + config['batch_size'])], Y_train[i:(
                i + config['batch_size'])]
        if config['cuda']:
            x, y = Variable(torch.from_numpy(x).type(torch.FloatTensor), requires_grad=True).cuda(), Variable(torch.from_numpy(y)).cuda()
        else:
            x, y = Variable(torch.from_numpy(x).type(torch.FloatTensor), requires_grad=True), \
                Variable(torch.from_numpy(y))

        # 将模型的参数梯度设置为0
        y = y.long()
        optimizer.zero_grad()
        output = model(x)
        pred = output.data.max(1, keepdim=True)[1]
        #print(pred.shape, y.shape)
        correct += pred.eq(y.data.view_as(pred)).cpu().sum()
        loss = criterion(output, y)
        # 后向传播调整参数
        loss.backward()
        # 根据梯度更新网络参数
        optimizer.step()
        batch_idx += 1
        # loss.item()能够得到张量中的元素值
        total_loss += loss.item()

    train_acc = correct
    avg_loss=total_loss/len(X_train)
    acc=correct / X_train.shape[0]
    print(
        'Train Epoch: {:2d}\t Learning rate: {:.4f}\tLoss: {:.6f}\t Accuracy: {}/{} ({:.0f}%)\n '
        .format(epoch + 1, config['learning_rate'], total_loss/len(X_train), correct,
                X_train.shape[0], 100. * correct / X_train.shape[0]))
    return avg_loss, acc


def evaluate(model, test_idxs, fold, train_idxs):
    model.eval()
    batch_idx = 1
    total_loss = 0
    global max_f1, max_acc, min_mae, X_test_lens, max_prec, max_rec
    pred = np.array([])
    with torch.no_grad():
        if config['cuda']:
            x, y = Variable(torch.from_numpy(text_features[test_idxs]).type(torch.FloatTensor), requires_grad=True).cuda(),\
                Variable(torch.from_numpy(text_targets[test_idxs])).cuda()
        else:
            x, y = Variable(torch.from_numpy(text_features[test_idxs]).type(torch.FloatTensor), requires_grad=True), \
                Variable(torch.from_numpy(text_targets[test_idxs])).type(torch.LongTensor)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        total_loss += loss.item()
        y_test_pred, conf_matrix = model_performance(y, output.cpu())
        accuracy = float(conf_matrix[0][0] + conf_matrix[1][1]) / np.sum(conf_matrix)
        precision = float(conf_matrix[0][0]) / (conf_matrix[0][0] + conf_matrix[0][1])
        recall = float(conf_matrix[0][0]) / (conf_matrix[0][0] + conf_matrix[1][0])
        f1_score = 2 * (precision * recall) / (precision + recall)
        print("Accuracy: {}".format(accuracy))
        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1-Score: {}\n".format(f1_score))
        print('=' * 89)

        if max_f1 <= f1_score and train_acc > len(train_idxs)*0.83 and f1_score > 0.7:
            max_f1 = f1_score
            max_acc = accuracy
            max_rec = recall
            max_prec = precision
            '''save(model, os.path.join(prefix,'Model/ClassificationWhole/Text/BiLSTM_{}_{:.2f}_{}'.format(config['hidden_dims'],max_f1, fold)))'''
            #save(model, os.path.join(prefix, 'Model/ClassificationWhole/Text/BiLSTM_{}_{:.2f}_{}'.format(config['hidden_dims'], max_f1, fold)))
            #save(model, os.path.join(prefix,'DAICCode/Model/BiLSTM_{}_{:.2f}_{}'.format(config['hidden_dims'],max_f1, fold)))
            print('*' * 64)
            print('model saved: f1: {}\tacc: {}'.format(max_f1, max_acc))
            print('*' * 64)

    return accuracy,precision

def get_param_group(model):
    nd_list = []
    param_list = []
    for name, param in model.named_parameters():
        if 'ln' in name:
            nd_list.append(param)
        else:
            param_list.append(param)
    return [{'params': param_list, 'weight_decay': 1e-5}, {'params': nd_list, 'weight_decay': 0}]

def pltshow(epoch,losses,accuracies,split='train'):
    if(split=='train'):
        epochs = range(1, epoch + 1)

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, losses, 'r', label=' loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, accuracies, 'b', label=' accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()
    else:
        epochs = range(1, epoch + 1)

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, losses, 'r', label=' precision')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, accuracies, 'b', label=' accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()



config = {
    'num_classes': 2,
    'dropout': 0.5,
    'rnn_layers': 2,
    'embedding_size': 1024,
    'batch_size': 8,
    'epochs': 70,
    'learning_rate': 8e-5,
    'hidden_dims': 128,
    'bidirectional': True,
    'cuda': False,
}

if __name__ == '__main__':
    kf = KFold(n_splits=3, shuffle=True)
    fold = 1
    print('1')
    for train_idxs_tmp, test_idxs_tmp in kf.split(text_features):
        # if idx_idx != 2:
        #     continue
        train_idxs, test_idxs = [], []
        # depression data augmentation
        for idx in train_idxs_tmp:
            if idx in text_dep_idxs_tmp:
                feat = text_features[idx]
                count = 0
                resample_idxs = [0, 1, 2, 3, 4, 5]
                for i in itertools.permutations(feat, feat.shape[0]):
                    if count in resample_idxs:
                        text_features = np.vstack((text_features, np.expand_dims(list(i), 0)))
                        text_targets = np.hstack((text_targets, 1))
                        train_idxs.append(len(text_features) - 1)
                    count += 1
            else:
                train_idxs.append(idx)

        for idx in test_idxs_tmp:

            if idx in text_dep_idxs_tmp:
                feat = text_features[idx]
                count = 0
                # resample_idxs = random.sample(range(6), 4)
                resample_idxs = [0, 1, 4, 5]
                for i in itertools.permutations(feat, feat.shape[0]):
                    if count in resample_idxs:
                        text_features = np.vstack((text_features, np.expand_dims(list(i), 0)))
                        text_targets = np.hstack((text_targets, 1))
                        test_idxs.append(len(text_features) - 1)
                    count += 1
            else:
                test_idxs.append(idx)
        print('2')
        model = TextBiLSTM(config)

        param_group = get_param_group(model)
        optimizer = optim.AdamW(param_group, lr=config['learning_rate'])
        criterion = nn.CrossEntropyLoss()
        max_f1 = -1
        max_acc = -1
        max_rec = -1
        max_prec = -1
        train_acc = -1

        losses_t = []
        accuracies_t = []
        precision_v = []
        accuracies_v = []
        print("begin")
        for ep in range(1, config['epochs'] + 1):

            loss_t, acc_t = train(ep, train_idxs)
            losses_t.append(loss_t)
            accuracies_t.append(acc_t)

            acc_v, precision = evaluate(model, test_idxs, fold, train_idxs)
            precision_v.append(precision)
            accuracies_v.append(acc_v)
        pltshow(config['epochs'], losses_t, accuracies_t, split='train')
        pltshow(config['epochs'], precision_v, accuracies_v, split='test')
        fold += 1

'''train_idxs_tmps = [np.load(os.path.join(prefix, 'Features/TextWhole/train_idxs_0.66_1.npy'), allow_pickle=True),
np.load(os.path.join(prefix, 'Features/TextWhole/train_idxs_0.72_2.npy'), allow_pickle=True),
np.load(os.path.join(prefix, 'Features/TextWhole/train_idxs_0.67_3.npy'), allow_pickle=True)]'''
