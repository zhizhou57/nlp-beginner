import pandas as pd
import numpy as np
import torch
import re
from torch.utils.data import Dataset, random_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import DataLoader

from model import TextCNN
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
class TextDataset(Dataset):
    def __init__(self, csv_path, label=False, num_classes=5):
        super().__init__()
        print("start reading text")
        # 读取数据
        data = pd.read_csv(csv_path, sep='\t')
        # 清除句子中不必要的成分
        self.x = [clean_text(x) for x in data['Phrase']]
        self.feature = None

        # 进行分词构建词汇表, 共16387个词
        print("start tokenizing the corpus")
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(self.x)
        print("finish tokenizing, totally {} words".format(len(self.tokenizer.word_index)))

        if label is not None:
            self.y = np.zeros((len(data['Sentiment']), num_classes))
            for i in range(len(self.y)):
                self.y[i][int(data['Sentiment'][i])] = 1

        # 提取出句子的feature
        self.feature_extract()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        '''
        返回一个句子的word embedding表示
        :param item:
        :return:
        '''
        if self.y is None:
            return torch.tensor(self.feature[item])
        else:
            return torch.tensor(self.feature[item]), torch.tensor(self.y[item])

    def feature_extract(self):
        # 获取最大句子长度:45
        max_len = np.max([len(text.split()) for text in self.x])
        print("max length of corpus is {}".format(max_len))

        # 对句子进行填充
        self.feature = self.tokenizer.texts_to_sequences(self.x)
        self.feature = pad_sequences(self.feature, maxlen=max_len)


def clean_text(documents: str, punctuations=r'''!()-[]{};:'"\,<>./?@#$%^&*_~''',
               stop_words=None) -> str:
    """
    # 清除句子中不必要的成分
    """
    # 移除标点符号
    if stop_words is None:
        stop_words = ['the', 'a', 'and', 'is', 'be', 'will']

    for x in documents.lower():
        if x in punctuations:
            documents = documents.replace(x, "")

    # 所有单词转为小写
    documents = documents.lower()

    # 移除停顿词
    documents = ' '.join([word for word in documents.split() if word not in stop_words])

    # 删除空格
    string = re.sub(r'\s+', ' ', documents).strip()

    return documents

vocab_size = 16400
embedding_size = 200
num_classes = 5
num_filters = 100
kernel_size = [3, 4, 5]
dropout_rate = 0.3
learning_rate = 0.001
epochs = 10

if __name__ == '__main__':
    path = "./data/train.tsv"
    textDataset = TextDataset(path, True)
    # 划分训练集、测试集
    train_dataset, test_dataset = random_split(textDataset, [int(len(textDataset)*0.8), int(len(textDataset)*0.2)])
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    print("size of training data: {}".format(len(train_dataset)))
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    print("size of testing data:{}".format(len(test_dataset)))

    # 构建训练参数
    model = TextCNN(vocab_size, embedding_size, num_classes, num_filters, kernel_size, dropout_rate).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        ep_loss = 0
        for step, data in enumerate(train_dataloader):
            input, label = data
            input = input.cuda()
            label = label.cuda()
            output = model(input)
            loss = loss_func(output, label)
            ep_loss += loss.item()

            model.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                print("Epoch: {}, Step: {}, Loss:{}".format(epoch, step, loss.item()))

        model.eval()
        with torch.no_grad():
            corr_num = 0
            err_num = 0
            for step, data in enumerate(test_dataloader):
                input, label = data
                input = input.cuda()
                label = label.cuda()
                output = model(input)
                corr_num += (output.argmax(1) == label.argmax(1)).sum().item()
                err_num += (output.argmax(1) != label.argmax(1)).sum().item()
            print("Epoch: {}, Accuracy: {}".format(epoch, corr_num / (corr_num + err_num)))