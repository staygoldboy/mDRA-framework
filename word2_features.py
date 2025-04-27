import re
import numpy as np
import pickle
import torch
from tqdm import tqdm
import os
import nltk
from jieba import tokenize
import jieba
import pandas as pd
prefix = os.path.abspath(os.path.join(os.getcwd(), "."))

def get_samples_from_text(path):
    f = pd.read_csv(path)
    sample_list = list(zip(f['content'], f['label']))
    return sample_list



class ChineseTensorEncoder():
    def __init__(self, vocab_path, dataset_name, datafile_path, sent_length: int, embedding_dim: int,
                  bias=3) -> None:
        super(ChineseTensorEncoder, self).__init__()
        self.vocab_path = vocab_path
        self.dataset_name = dataset_name
        self.datafile_path = datafile_path
        self.sent_length = sent_length
        self.embedding_dim = embedding_dim
        self.bias = bias

    def encode(self):
        text_targets=[]
        word_embedding_dict = {}
        word_embedding_list = []
        with open(self.vocab_path, "r", encoding='utf-8') as f:
            word_count = 0
            for line in f:
                word_count += 1
                if (word_count == 1):
                    continue
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], "float16")
                word_embedding_dict[word] = vector
                word_embedding_list.append(vector.tolist())
        print(np.array(word_embedding_list).shape)
        mean_embedding = np.mean(np.array(list(word_embedding_dict.values())), axis=0)
        zero_embedding = np.array([0] * self.embedding_dim, dtype=float)
        mean_value = np.mean(np.array(word_embedding_list))
        variance_value = np.var(np.array(word_embedding_list))
        print(mean_value)
        print(variance_value)
        left_boundary = mean_value - self.bias * np.sqrt(variance_value)
        right_boundary = mean_value + self.bias * np.sqrt(variance_value)

        sample_list = get_samples_from_text(path=self.datafile_path)

        embedding_tuple_list = []
        for i in tqdm(range(len(sample_list))):
            sent_embedding = np.array([[0] * self.embedding_dim] * self.sent_length, dtype=float)
            text_list = list(jieba.tokenize(sample_list[i][0]))
            label = sample_list[i][1]
            for j in range(self.sent_length):
                if j >= len(text_list):
                    embedding_norm = zero_embedding  # zero padding
                else:
                    word = text_list[j][0]
                    embedding = word_embedding_dict[word] if word in word_embedding_dict.keys() else zero_embedding
                    # N(0, 1)
                    embedding_n01 = (embedding - np.array([mean_value] * self.embedding_dim)) / np.array(
                        [np.sqrt(variance_value)] * self.embedding_dim)
                    embedding_norm = np.array([0] * self.embedding_dim, dtype=float)
                    for k in range(self.embedding_dim):
                        if embedding[k] < left_boundary:
                            embedding_norm[k] = -self.bias
                        elif embedding[k] > right_boundary:
                            embedding_norm[k] = self.bias
                        else:
                            embedding_norm[k] = embedding_n01[k]
                    # add abs(left_embedding)
                    embedding_norm = (embedding_norm + np.array([np.abs(self.bias)] * self.embedding_dim)) / (
                                self.bias * 2)
                    # embedding_norm = np.clip(embedding_norm, a_min=0, a_max=1)
                sent_embedding[j] = embedding_norm
            embedding_tuple_list.append(sent_embedding)
            text_targets.append(label)

        merged_list = []
        text_label = []
        # 按照每3条数据合并
        for i in range(0, len(embedding_tuple_list), 3):
            data_group = embedding_tuple_list[i:i + 3]
            # 计算这3个数组的均值
            mean_data = np.mean(data_group, axis=0)
            # 将计算出的均值数组添加到新的数据列表中
            merged_list.append(mean_data)

        for i in range(0, len(sample_list), 3):
            data_group = [sample_list[i][1], sample_list[i + 1][1], sample_list[i + 2][1]]
            mean_data = np.mean(data_group, axis=0)
            text_label.append(mean_data)
        np.savez(os.path.join(prefix, 'Features/TextWhole/whole_samples_300_word2.npz'), merged_list)
        np.savez(os.path.join(prefix, 'Features/TextWhole/whole_labels_300_word2.npz'), text_label)





if __name__ == "__main__":
    tensor_encoder = ChineseTensorEncoder(
        vocab_path="D:/pywork/snn-covert/snn/word2vec/sgns.target.word-ngram.1-2.dynwin5.thr10.neg5.dim300.iter5",
        dataset_name="copus",
        datafile_path="D:/pywork/icassp2022-depression-main/DepressionCollected/Classification/copus.csv",
        sent_length=3,
        embedding_dim=300,
        bias=3
    )
    tensor_encoder.encode()