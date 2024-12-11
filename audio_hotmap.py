import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from sklearn.metrics import f1_score, recall_score, precision_score

prefix = os.path.abspath(os.path.join(os.getcwd(), "."))

image_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}
# 三、加载数据
# torchvision.transforms包DataLoader是 Pytorch 重要的特性，它们使得数据增加和加载数据变得非常简单。
# 使用 DataLoader 加载数据的时候就会将之前定义的数据 transform 就会应用的数据上了。

train_directory = os.path.join(prefix, 'heatmap1/train')
valid_directory = os.path.join(prefix, 'heatmap1/valid')

batch_size = 16
num_classes = 2
print(train_directory)
data = {
    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid'])
}

train_data_size = len(data['train'])
valid_data_size = len(data['valid'])

train_data = DataLoader(data['train'], batch_size=batch_size, shuffle=True, num_workers=0)
valid_data = DataLoader(data['valid'], batch_size=batch_size, shuffle=True, num_workers=0)

print(train_data_size, valid_data_size)


class BinaryClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(BinaryClassifier, self).__init__()
        # 加载预训练的ResNet50模型
        self.resnet = models.resnet50(pretrained=True)
        # 冻结特征提取层
        for param in self.resnet.parameters():
            param.requires_grad = False
        # 修改最后的全连接层
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = BinaryClassifier().to(device)

# 定义损失函数和优化器。
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.resnet.fc.parameters(),lr=0.001)

# 五、训练
def train_and_valid(model, loss_function, optimizer, epochs=25):

    history = []
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))

        model.train()

        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
        all_predictions = []
        all_labels = []

        for i, datas in enumerate(tqdm(train_data)):
            inputs,labels = datas
            inputs = torch.tensor(inputs)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 因为这里梯度是累加的，所以每次记得清零
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            train_acc += acc.item() * inputs.size(0)

        with torch.no_grad():
            model.eval()

            for j, (inputs, labels) in enumerate(tqdm(valid_data)):
                inputs = torch.tensor(inputs)
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)
                ret, predictions = torch.max(outputs.data, 1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                valid_acc += acc.item() * inputs.size(0)

        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size

        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_acc / valid_data_size
        f1 = f1_score(all_labels, all_predictions, average='macro')
        recall = recall_score(all_labels, all_predictions, average='macro')
        precision = precision_score(all_labels, all_predictions, average='macro')
        print(f'F1 Score: {f1}')
        print(f'Recall: {recall}')
        print(f'Precision: {precision}')
        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        if best_acc < avg_valid_acc:
            best_acc = avg_valid_acc
            best_epoch = epoch + 1

        epoch_end = time.time()

        print(
            "Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
                epoch + 1, avg_valid_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100,
                epoch_end - epoch_start
            ))
        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))

        #torch.save(model, 'models/' + dataset + '_model_' + str(epoch + 1) + '.pt')
    return model, history


num_epochs = 30
trained_model, history = train_and_valid(model, loss_func, optimizer, num_epochs)
#torch.save(history, 'models/' + dataset + '_history.pt')

history = np.array(history)
plt.plot(history[:, 0:2])
plt.legend(['Tr Loss', 'Val Loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.ylim(0, 1)
#plt.savefig(dataset + '_loss_curve.png')
plt.show()

'''plt.plot(history[:, 2:4])
plt.legend(['Tr Accuracy', 'Val Accuracy'])
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
#plt.savefig(dataset + '_accuracy_curve.png')
plt.show()'''

plt.plot(history[:, 3:4])
plt.legend(['Val Accuracy'])
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
#plt.savefig(dataset + '_accuracy_curve.png')
plt.show()
