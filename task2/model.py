import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_classes=5, num_filters=100, kernel_sizes=None, dropout_rate=0.3):
        super(TextCNN, self).__init__()
        # vocab_size:总词数， embedding_size:映射到高维空间的维度
        if kernel_sizes is None:
            kernel_sizes = [3, 4, 5]

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # 卷积层
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_size), padding=(k-1, 0))
            for k in kernel_sizes])
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def conv_pool(self, x, conv):
        x = F.relu(conv(x).squeeze(3))  # (batch_size, num_filter, conv_seq_length)
        x_max = F.max_pool1d(x, x.size(2)).squeeze(2)  # (batch_size, num_filter)
        return x_max

    def forward(self, x):
        embedding = self.embedding(x).unsqueeze(1).cuda()

        conv_results = [self.conv_pool(embedding, conv) for conv in self.convs]
        out = torch.cat(conv_results, 1).cuda()
        out = self.dropout(out)
        return self.fc(out)

