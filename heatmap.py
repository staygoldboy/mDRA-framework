import os
import matplotlib.pyplot as plt
import librosa
import numpy as np
from librosa import feature
import math


prefix = os.path.abspath(os.path.join(os.getcwd(), "."))


def extract_mfcc_features(wav_file):
    # 加载音频文件
    y, sr = librosa.load(wav_file,sr=None)

    # 计算MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # 计算对数能量
    sr1=int(sr)
    log_energy = librosa.feature.rms(y=y, frame_length=sr1)

    # 将对数能量添加到MFCC矩阵
    features = np.vstack([log_energy, mfcc])
    # 计算均值、一阶导数和二阶导数
    mean = np.mean(features, axis=1)

    delta1 = librosa.feature.delta(mean, order=1)
    delta2 = librosa.feature.delta(mean, order=2)
    # 计算标准差、一阶导数和二阶导数
    std_dev = np.std(features, axis=1)

    delta1_std = librosa.feature.delta(std_dev, order=1)
    delta2_std = librosa.feature.delta(std_dev, order=2)

    # 计算基频值
    # 将音频划分成14个等长的片段
    num_segments = 14
    segment_length = len(y) // num_segments
    f3 = []

    for i in range(num_segments):
        segment_start = i * segment_length
        segment_end = (i + 1) * segment_length
        segment = y[segment_start:segment_end]

        # 计算基频
        f0, voiced_flag, voiced_probs = librosa.pyin(segment, fmin=librosa.note_to_hz('C2'),fmax=librosa.note_to_hz('C7'))
        # 使用列表推导式过滤空值
        f = [value for value in f0 if value is not None and not math.isnan(value)]

        f2 = np.mean(f).astype(np.float32)
        f3.append(f2)
    f3 = np.array(f3)
    # 将所有特征值组合成一个7x14矩阵
    mfcc_feature_matrix = np.vstack([f3, mean, delta1, delta2, std_dev, delta1_std, delta2_std])

    return mfcc_feature_matrix

def plot_heatmap(matrix, output_path):
    plt.figure(figsize=(10, 6))
    plt.imshow(matrix, cmap='viridis', aspect='auto', origin='lower')
    #plt.colorbar(label='MFCC Coefficients')
    plt.xlabel('MFCC Coefficients')
    plt.ylabel('Features')
    plt.title('MFCC Feature Heatmap')

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print('完成')
    plt.close()

a=20
b=65

for index in range(114):
    path = os.path.join(prefix, 'EATD-Corpus/v_{}/combined_audio.wav'.format(index+1))
    if not os.path.exists(path):
        print('错误')
    else:
        mfcc_features = extract_mfcc_features(path)
        with open(os.path.join(prefix, 'EATD-Corpus/v_{}/new_label.txt'.format(index+1))) as fli:
            target = float(fli.readline())
        if target >= 53.0:
            pngpath = os.path.join(prefix, 'heatmap1/1/{}.png'.format(a))
            plot_heatmap(mfcc_features,pngpath)
            a+=1
        else:
            pngpath = os.path.join(prefix, 'heatmap1/0/{}.png'.format(b))
            plot_heatmap(mfcc_features, pngpath)
            b+=1
