import os
import time
import glob
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import load_data, accuracy
from models import GAT

from sklearn import manifold, datasets
from visdom import Visdom

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='不使用GPU训练')
parser.add_argument('--fastmode', action='store_true', default=False, help='训练时进行验证')
parser.add_argument('--sparse', action='store_true', default=False, help='是否使用稀疏存储')
parser.add_argument('--epochs', type=int, default=10000, help='迭代数')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.005, help='初始学习率')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='L2衰减权重')
parser.add_argument('--hidden', type=int, default=8, help='隐藏层维度')
parser.add_argument('--nb_heads', type=int, default=8, help='attention heads数')
parser.add_argument('--dropout', type=float, default=0.6, help='丢失率')
parser.add_argument('--alpha', type=float, default=0.2, help='leaky_relu系数')
parser.add_argument('--patience', type=int, default=100, help='等待迭代')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# Model and optimizer
model = GAT(nfeat=features.size()[1],
            nhid=args.hidden,
            nclass=int(labels.max())+1,
            dropout=args.dropout,
            nheads=args.nb_heads,
            alpha=args.alpha)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

features, adj, labels = Variable(features), Variable(adj), Variable(labels)


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch：{:04d}'.format(epoch+1),
          'loss_train：{:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item()


def compute_test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data.item()),
          "accuracy= {:.4f}".format(acc_test.data.item()))
    return output

# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0

for epoch in range(args.epochs):
    loss_values.append(train(epoch))

    torch.save(model.state_dict(), '{}.pkl'.format(epoch))
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

    files = glob.glob('*.pkl')  # 查询当前目录下后缀为.pkl的文件名
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)

files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("总用时：{:.4f}s".format(time.time() - t_total))

# 读模型，用于测试
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

# 测试
output = compute_test()

# output的格式转换
output = output.cpu().detach().numpy()
labels = labels.cpu().detach().numpy()


# t-SNE 降维
def t_SNE(output, dimention):
    """
    :param output:待降维的数据
    :param dimention：降低到的维度
    """
    tsen = manifold.TSNE(n_components=dimention,
                         init='pca',
                         random_state=0)
    result = tsen.fit_transform(output)
    return result


# Visualization with visdom
def Visualization(result, labels):
    vis = Visdom()
    vis.scatter(
        X=result,
        Y=labels + 1,  # 将label的最小值从0变为1，显示时label不可为0
        opts=dict(markersize=5, title='Dimension reduction to %dD' %(result.shape[1])),
    )


# result = t_SNE(output, 2)
# Visualization(result, labels)

result = t_SNE(output, 3)
# result = np.load('3维数据.npy')
Visualization(result, labels)









