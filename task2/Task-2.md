# Task-2

## 1.文本的特征表示

### Embedding层

望文生义，是将原本字符串形式的词汇，嵌入到一个高维空间（维度embedding_size）中，使得不同的词汇间具有不同的距离，

具体实现方式：word2vec(包括Skip-gram, CBOW)，GloVe

### word2vec

CBOW或Skip-gram，本实验采用随机embedding

### GloVe

Stanford的预训练模型，因个人电脑显存不够，因此暂未采用

## 2.TextCNN

- embedding层
- 一维卷积层（kernelsize为k的一维卷积等价于kernelsize为（k， embedding）的二维卷积）
- 采用多个kernel size的卷积，结果连接到一起（类似n-gram？）
- 接最大池化层
- 接全连接层（之前dropout防止过拟合）

## 3.文本分类实现

1. 预处理得到所有的词汇，并进行embedding
2. 对于每句话，其feature是其中所有词汇Vector值组成的高维narray，长度取所有句子包含词汇的最大值，不够则填充0
3. 构建数据集验证集
4. 构建CNN模型/RNN模型进行训练
5. 调参，获取好的结果（随机embedding,粗略调参，准确度65%）

### 未解决的问题

- torch.nn.functional与torch.nn的区别
- 什么时候将模型、数据加载到GPU
- 定量调参并画图