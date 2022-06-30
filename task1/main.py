import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class Text_dataset():
    def __init__(self, path, max_feature):
        '''
        :param path: 数据路径
        :maxfeature: n-gram中的最大特征数
        '''
        data_all = pd.read_csv(path, sep='\t')
        self.x = data_all['Phrase']
        if "Sentiment" in data_all.columns:
            # 此时为测试集
            self.y = data_all['Sentiment']
        print("total sentence: {}".format(self.x.shape[0]))
        self.max_feature = max_feature

    # def __getitem__(self, item):
    #     X = self.x[item]
    #     if self.y is not None:
    #         # 训练集
    #         Y = self.y[item]
    #         return X, Y
    #     else:
    #         # 测试集
    #         return X

    def feature_extract(self, n, tfidf=False):
        '''
        提取文本的特征
        :param n: n-gram 中n的值，若为1则为bag-of-word
        :param tfidf: 是否采用tf-idf特征
        :return:
        '''
        if not tfidf:
            ngram_vect = CountVectorizer(ngram_range=(1, n))
            self.x_ngram = ngram_vect.fit_transform(self.x)
        else:
            ngram_vect_tfidf = TfidfVectorizer(ngram_range=(1, n))
            self.x_ngram = ngram_vect_tfidf.fit_transform(self.x)
        print("total feature size:{}".format(self.x_ngram.shape))
        return self.x_ngram, self.y


def train(train_x, train_y, classifier):
    print("training data size:{}, label size: {}".format(train_x.shape, train_y.shape))
    train_x, train_y = shuffle(train_x, train_y)
    classifier.fit(train_x, train_y)
    return classifier


def predict(valid_x, valid_y, classifier):
    predict_y = classifier.predict(valid_x)
    print("validation accuracy:{}".format(np.mean(predict_y == valid_y)))


# path = "./data"
# train_path = os.path.join(path, "train.tsv")
# test_path = os.path.join(path, "test.tsv")
root = os.getcwd()
train_path = "data\\train.tsv"
test_path = "data\\test.tsv"

parser = argparse.ArgumentParser()
parser.add_argument("-max_n", default=2, type=int, help="size of n-gram model")
parser.add_argument("-max_feature", default=None, type=int, help="max size of n-gram model")
parser.add_argument("-classifier", default="SGD", help="which classifier to choose")
parser.add_argument("-alpha", default=0.001, type=float, help="SGD alpha")
parser.add_argument("-max_iter", default=1000, type=int, help="SGD iteration times")

if __name__ == '__main__':

    args = parser.parse_args()
    max_n = args.max_n
    max_feature = args.max_feature
    print("Reading training data")
    data = Text_dataset(train_path, max_feature)

    classifier = args.classifier
    if classifier == "softmax":
        clf = LogisticRegression(random_state=0, multi_class='multinomial')
    elif classifier == "SGD":
        alpha = args.alpha
        max_iter = args.max_iter
        clf = SGDClassifier(alpha=alpha, loss='log', early_stopping=True, eta0=0.001, learning_rate='adaptive',
                            max_iter=max_iter)
    else:
        print("No such classifier! Exit")
        sys.exit(1)

    feature, label = data.feature_extract(max_n, True)
    train_x, valid_x, train_y, valid_y = train_test_split(feature, label, test_size=0.2)

    clf = train(train_x, train_y, clf)
    predict(valid_x, valid_y, clf)
    # print("Reading test data")
    # test_data = Text_dataset(test_path)
