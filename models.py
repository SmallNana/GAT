import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, GATLayer


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(in_features=nfeat,
                                               out_features=nhid,
                                               dropout=dropout,
                                               alpha=alpha,
                                               concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(in_features=nhid * nheads,
                                           out_features=nclass,
                                           dropout=dropout,
                                           alpha=alpha,
                                           concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        return F.log_softmax(x, dim=1)


class GAT_ppi(nn.Module):
    """GAT for ppi"""
    def __init__(self, num_of_layers, num_heads_per_layer, num_features_per_layer,
                 add_skip_connection=True, bias=True, dropout=0.6, log_attention_weights=False):
        """
        :param num_of_layers: 3
        :param num_head_per_layer: [4,4,6]
        :param num_features_per_layer: [50, 64, 64, 121]
        :param add_skip_connection: True
        :param bias:True
        :param dropout:0.0
        :param log_attention_weights:False 是否保存attention数据，这是用来可视化的
        """
        super().__init__()
        assert num_of_layers == len(num_heads_per_layer) == len(num_features_per_layer) - 1, f'Enter valid arch params.'
        num_heads_per_layer = [1] + num_heads_per_layer

        gat_layers = []
        for i in range(num_of_layers):
            layer = GATLayer(
                num_in_features=num_heads_per_layer[i] * num_features_per_layer[i],
                num_out_features=num_features_per_layer[i + 1],
                num_of_heads=num_heads_per_layer[i+1],
                concat=True if i < num_of_layers-1 else False,
                activation=nn.ELU() if i < num_of_layers-1 else None,
                dropout_prob=dropout,
                add_skip_connection=add_skip_connection,
                bias=bias,
                log_attention_weights=log_attention_weights
            )
            gat_layers.append(layer)

        """
         nn.Sequential一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，同时以神经网络模块为元素的有序字典也可以作为传入参数。
         nn.ModuleList,仅仅类似于pytho中的list类型，只是将一系列层装入列表，并没有实现forward()方法,因此也不会有网络模型产生的副作用。
         nn.ModuleList接受的必须是subModule类型，即不管ModuleList包裹了多少个列表，内嵌的所有列表的内部都要是可迭代的Module的子类
         不同点1: nn.Sequential内部实现了forward函数，因此可以不用写forward函数。而nn.ModuleList则没有实现内部forward函数。
         不同点2：nn.Sequential可以使用OrderedDict对每层进行命名
         不同点3：nn.Sequential里面的模块按照顺序进行排列的，所以必须确保前一个模块的输出大小和下一个模块的输入大小是一致的。
                 而nn.ModuleList 并没有定义一个网络，它只是将不同的模块储存在一起，这些模块之间并没有什么先后顺序可言。
         不同点4：有的时候网络中有很多相似或者重复的层，我们一般会考虑用 for 循环来创建它们，而不是一行一行地写
        """

        self.gat_net = nn.Sequential(
            *gat_layers,
        )

    def forward(self, data):
        return self.gat_net(data)