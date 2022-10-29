import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import copy
import math


class AGGCN(nn.Module):
    def __init__(self, opt):
        """
        Attention Guided GCN
        :param opt: NameSpace<- arg_parser
        """
        super(AGGCN, self).__init__()
        self.opt = opt
        self.in_dim = opt['in_dim']
        self.use_cuda = torch.cuda.is_available()

        self.mem_dim = opt['hidden_dim']
        self.in_drop = nn.Dropout(opt['input_dropout'])
        self.num_layers = opt['num_layers']

        self.layers = nn.ModuleList()
        self.heads = opt['heads']
        self.sublayer_first = opt['sublayer_first']
        self.sublayer_second = opt['sublayer_second']

        # gcn layer
        for i in range(self.num_layers):
            if i == 0:
                # 先将普通的邻接矩阵 过两个 densely connection layer -- DCL
                # 第一个DCL为2层 第二个DCL为4层
                self.layers.append(GraphConvLayer(opt, self.mem_dim, self.sublayer_first))
                self.layers.append(GraphConvLayer(opt, self.mem_dim, self.sublayer_second))
            else:
                # 再执行多头注意力 + DCL
                self.layers.append(MultiGraphConvLayer(opt, self.mem_dim, self.sublayer_first, self.heads))
                self.layers.append(MultiGraphConvLayer(opt, self.mem_dim, self.sublayer_second, self.heads))
        # 每一个DCL的输出为维度都是 mem_dim维
        self.aggregate_W = nn.Linear(len(self.layers) * self.mem_dim, self.mem_dim)

        self.attn = MultiHeadAttention(self.heads, self.mem_dim)
    def forward(self, adj, inputs, word_sample_mask):
        word_rep = inputs
        # batch_size * max_word_len  -> batch_size * 1 * max_word_len
        src_mask = (word_sample_mask != 0).unsqueeze(-2)
        gcn_inputs = self.in_drop(word_rep)
        layer_list = []
        outputs = gcn_inputs
        for i in range(len(self.layers)):
            if i < 2:
                # 邻接矩阵直接进入 dcl
                # adj 50 * n * n     50 *n * 300
                outputs = self.layers[i](adj, outputs)
                layer_list.append(outputs)
            else:
                # batch_size * h * n * n
                attn_tensor = self.attn(outputs, outputs, src_mask)
                attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]
                outputs = self.layers[i](attn_adj_list, outputs)
                layer_list.append(outputs)

        aggregate_out = torch.cat(layer_list, dim=2)
        dcgcn_output = self.aggregate_W(aggregate_out)
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)

        return dcgcn_output, mask

class GraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, opt, mem_dim, layers):
        """

        :param opt: config 配置dict
        :param mem_dim: hidden_size - 300
        :param layers: 多少层， 第一个DCL是2层， 第二个DCL是4层
        """
        super(GraphConvLayer, self).__init__()
        # config dict
        self.opt = opt
        # hidden_size == 768
        self.mem_dim = mem_dim
        # 层数
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.gcn_drop = nn.Dropout(opt['gcn_dropout'])

        # dcgcn layer   densly connection layer
        self.Linear = nn.Linear(self.mem_dim, self.mem_dim)
        self.weight_list = nn.ModuleList()
        for i in range(self.layers):
            # 根据层数来确定每层的模型
            # 第一层 768 -> 384  第二层 768+384 -> 384
            self.weight_list.append(nn.Linear((self.mem_dim + self.head_dim * i), self.head_dim))

        if torch.cuda.is_available():
            self.weight_list = self.weight_list.cuda()
            self.Linear = self.Linear.cuda()

    def forward(self, adj, gcn_inputs):
        """

        :param adj: 50 * max_len * max_len
        :param gcn_inputs: 图神经网络的输入，， batch_size * max_len * hidden_size
        :return:
        """
        denom = adj.sum(2).unsqueeze(2) + 1

        outputs = gcn_inputs
        cache_list = [outputs]
        output_list = []

        for l in range(self.layers):
            Ax = adj.bmm(outputs)
            # self.weight_list[l] -> nn.Linear
            AxW = self.weight_list[l](Ax)
            AxW = AxW + self.weight_list[l](outputs)  # self loop
            AxW = AxW / denom
            gAxW = F.relu(AxW)  # 加激活函数 relu
            cache_list.append(gAxW)
            # 给下一层的输入做准备，，，下一层的输入为初始输入 拼接 前面几层的输出
            outputs = torch.cat(cache_list, dim=2)
            output_list.append(self.gcn_drop(gAxW))
        gcn_ouputs = torch.cat(output_list, dim=2)
        gcn_ouputs = gcn_ouputs + gcn_inputs

        out = self.Linear(gcn_ouputs)

        return out


class MultiGraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, opt, mem_dim, layers, heads):
        super(MultiGraphConvLayer, self).__init__()
        self.opt = opt
        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.heads = heads
        self.gcn_drop = nn.Dropout(opt['gcn_dropout'])

        # dcgcn layer
        self.Linear = nn.Linear(self.mem_dim * self.heads, self.mem_dim)
        self.weight_list = nn.ModuleList()

        for i in range(self.heads):
            for j in range(self.layers):
                self.weight_list.append(nn.Linear(self.mem_dim + self.head_dim * j, self.head_dim))

        if torch.cuda.is_available():
            self.weight_list = self.weight_list.cuda()
            self.Linear = self.Linear.cuda()

    def forward(self, adj_list, gcn_inputs):
        """

        :param adj_list: batch_szie * 1 * max_len *len
        :param gcn_inputs: batch_size * max_len * hidden_size
        :return:
        """
        multi_head_list = []
        for i in range(self.heads):
            adj = adj_list[i]
            denom = adj.sum(2).unsqueeze(2) + 1
            outputs = gcn_inputs
            cache_list = [outputs]
            output_list = []
            for l in range(self.layers):
                index = i * self.layers + l
                Ax = adj.bmm(outputs)
                AxW = self.weight_list[index](Ax)
                AxW = AxW + self.weight_list[index](outputs)  # self loop
                AxW = AxW / denom
                gAxW = F.relu(AxW)
                cache_list.append(gAxW)
                outputs = torch.cat(cache_list, dim=2)
                output_list.append(self.gcn_drop(gAxW))

            gcn_ouputs = torch.cat(output_list, dim=2)
            gcn_ouputs = gcn_ouputs + gcn_inputs

            multi_head_list.append(gcn_ouputs)

        final_output = torch.cat(multi_head_list, dim=2)
        out = self.Linear(final_output)

        return out


def attention(query, key, mask=None, dropout=None):
    """

    :param query: batch_size * h * n * d_k
    :param key:
    :param mask: batch_size * n
    :param dropout:
    :return:
    """
    d_k = query.size(-1)
    # batch_size * h * n * n
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        """
        多头注意力机制，生成h个 n*n的矩阵
        :param h: 头数
        :param d_model: 输入的input dim 为300维
        :param dropout:
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, mask=None):
        """
        输入两个一样的n*d_model的矩阵， 输出你*n
        :param query: b * n * d_model
        :param key:  b * n * d_model
        :param mask:  b * n
        :return:
        """
        if mask is not None:
            # mask b*n -> b*1*n
            mask = mask.unsqueeze(1)

        # batch_size
        nbatches = query.size(0)

        # batch_size * n * model_d   ->  batch_size * n * h * d_k (h*d_k==model_d) -> batch_size * h * n * d_k
        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]
        # query = query.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        # key = key.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        attn = attention(query, key, mask=mask, dropout=self.dropout)

        return attn
