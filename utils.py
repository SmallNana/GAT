import numpy as np
import scipy.sparse as sp
import torch
import random


def encode_onehot(labels):
    classes = set(labels)
    """获取标签对应的 onehot 编码映射关系"""
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}

    """获得各个节点标签的ont-hot编码"""
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def normalize_features(mx):
    """Row-normalize sparse matrix，row归一化，其实就是每一行归一化"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(mx):
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def load_data(path="data/cora/", dataset="cora"):
    print('loading {} dataset...'.format(dataset))

    """处理特征"""
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)

    """处理标签"""
    labels = encode_onehot(idx_features_labels[:, -1])  # one-hot编码

    """建图"""
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)  # 各个节点的id
    idx_map = {j: i for i, j in enumerate(idx)}  # 建立顶点和编号的映射  id: 顶点号
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)  # 读取论文引用关系
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    """将adj转变为对称矩阵"""
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    """归一化"""
    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_train = random.sample(range(0, 140), 140)  # 训练集索引
    idx_val = random.sample(range(140, 1708), 500)  # 验证集索引
    idx_test = random.sample(range(1708, 2708), 1000)  # 测试集索引

    """转为tensor表示"""
    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

