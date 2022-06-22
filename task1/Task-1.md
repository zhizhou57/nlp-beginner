# Task-1

## 1.文本的特征表示

把不方便直接处理的变长的文本，转换成方便处理的定长的向量

### Bag-of-word Model

1. Vocabulary or corpus(语料库): 在所有句子中出现过的单词（去重）
2. Count(计数)：计算每个句子中各个单词出现的频数，便是该句子的vector表示

例如：

| Document | the | cat | sat | in | hat | width |
| --- | --- | --- | --- | --- | --- | --- |
| the cat sat | 1 | 1 | 1 | 0 | 0 | 0 |
| the cat sat in the hat | 2 | 1 | 1 | 1 | 1 | 0 |
| the cat with the hat | 2 | 1 | 0 | 0 | 1 | 1 |

方便表示但会丢失句子的上下文信息。

### N-grams Model

构建一个大小为N的滑动窗口，统计每个N-gram出现的次数，如

**the cat sat**

**the cat sat in the hat**

**the cat with the hat**

其2-gram可以表示为：

|  | the | cat | sat | in | hat | with |
| --- | --- | --- | --- | --- | --- | --- |
| the | 0 | 3 | 0 | 0 | 2 | 0 |
| cat | 0 | 0 | 2 | 0 | 0 | 1 |
| sat | 0 | 0 | 0 | 1 | 0 | 0 |
| in | 1 | 0 | 0 | 0 | 0 | 0 |
| hat | 0 | 0 | 0 | 0 | 0 | 0 |
| with | 1 | 0 | 0 | 0 | 0 | 0 |

代码中采用scikit-learn中的CountVectorizer类实现

bag of words就是n等于1的特殊情况

### TF-IDF

用于统计一段文章中不同词的重要程度

TF（词频）= 该词的出现次数/文章的总词数

IDF（逆文档频率）= log(语料库的文档总数/包含该词的文档数+1)

TF-IDF=TF(词频)*IDF(逆文档频率)

## 2. 文本分类实现

1. 利用词袋模型或者n-gram模型，表示出句子的feature
2. 用shuttle划分训练集、验证集，也可采用N-fold
3. 将feature、label丢给softmax或SGD去训练
4. 调参，获取较好的结果（未细调，精确度大概55%）

### 未解决的问题

- 除了TF-IDF、n-gram有没有其他的文本特征适用于分类
- 调参绘图