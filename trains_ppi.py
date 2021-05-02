import argparse
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import f1_score

from models import GAT_ppi
from utils.constants import *
from utils.data_loading import load_graph_data
import utils.utils as utils


def get_training_args():
    """获得训练配置参数"""
    parser = argparse.ArgumentParser()

    # Training related
    parser.add_argument("--num_of_epochs", type=int, help="训练代数", default=200)
    parser.add_argument("--patience_period", type=int,
                        help="number of epochs with no improvement on val before terminating", default=100)
    parser.add_argument("--lr", type=float, help="模型初始学习率", default=5e-3)
    parser.add_argument("--weight_decay", type=float, help="L2 regularization 衰减率", default=0)
    parser.add_argument("--should_test", type=bool, help="是否测试", default=True)
    parser.add_argument("--force_cpu", type=bool, help="是否用cpu", default=False)

    # Dataset related
    parser.add_argument("--dataset_name", choices=[el.name for el in DatasetType],
                        help="选择数据集", default=DatasetType.PPI.name)
    parser.add_argument("--batch_size", type=int, help='一个batch有几张图', default=2)
    parser.add_argument("--should_visualize", type=bool, help="是否可视化数据集", default=False)

    # Logging/debugging/checkpoint related (helps a lot with experimentation)
    parser.add_argument("--enable_tensorboard", type=bool, help="enable tensorboard logging", default=False)
    parser.add_argument("--console_log_freq", type=int, help="log to output console (epoch) freq (None for no logging)",
                        default=10)
    parser.add_argument("--checkpoint_freq", type=int,
                        help="checkpoint model saving (epoch) freq (None for no logging)", default=5)
    args = parser.parse_args('')

    # 模型配置
    gat_config = {
        "num_of_layers": 3,  # PPI has got 42% of nodes with all 0 features - that's why 3 layers are useful
        "num_heads_per_layer": [4, 4, 6],  # other values may give even better results from the reported ones
        "num_features_per_layer": [PPI_NUM_INPUT_FEATURES, 64, 64, PPI_NUM_CLASSES],  # 64 would also give ~0.975 uF1!
        "add_skip_connection": True,  # skip connection is very important! (keep it otherwise micro-F1 is almost 0)
        "bias": True,  # bias doesn't matter that much
        "dropout": 0.0,  # dropout hurts the performance (best to keep it at 0)
    }

    # 把训练配置包装到字典中
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)
    training_config['ppi_load_test_only'] = False

    # 增加模型配置加入训练配置
    training_config.update(gat_config)

    return training_config


def get_main_loop(config, gat, sigmoid_cross_entropy_loss, optimizer, patience_period, time_start):
    """
    模型计算传递主过程，提高代码复用率
    :param config:配置参数
    :param gat:模型
    :param sigmoid_cross_entropy_loss:损失函数
    :param optimizer:优化器
    :param patience_period:最大等待代数
    :param time_start:开始时间
    """
    device = next(gat.parameters()).device  # 从模型中获取设备信息

    def main_loop(phase, data_loader, epoch=0):
        global BEST_VAL_MICRO_F1, BEST_VAL_LOSS, PATIENCE_CNT, writer

        if phase == LoopPhase.TRAIN:
            gat.train()
        else:
            gat.eval()

        for batch_idx, (node_features, gt_node_labels, edge_index) in enumerate(data_loader):
            """迭代一批图形数据，原论文是2张图，这里将2张图合为一张图，相当于一张图2个连通分量"""

            edge_index = edge_index.to(device)
            node_features = node_features.to(device)
            gt_node_labels = gt_node_labels.to(device)

            graph_data = (node_features, edge_index)  # 打包数据

            nodes_unnormalized_scores = gat(graph_data)[0]  # 最后输出的分数，还没经过Sigmoid，由于对于每个分量而言为2分类问题(0或1)，所以使用sigmoid

            loss = sigmoid_cross_entropy_loss(nodes_unnormalized_scores, gt_node_labels)

            if phase == LoopPhase.TRAIN:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 计算f1
            pred = (nodes_unnormalized_scores > 0).float().cpu().numpy()  # 只要得分大于0 sigmoid之后就大于0.5，那么就认为它是1
            gt = gt_node_labels.cpu().numpy()
            micro_f1 = f1_score(gt, pred, average='micro')

            # 记录数据

            global_step = len(data_loader) * epoch + batch_idx
            if phase == LoopPhase.TRAIN:
                # 记录指标
                if config['enable_tensorboard']:
                    writer.add_scalar('training_loss', loss.item(), global_step)
                    writer.add_scalar('training_micro_f1', micro_f1, global_step)

                # 记录数据在控制台，每代记录一次，记录的是这一代第一个batch
                if config['console_log_freq'] is not None and batch_idx % config['console_log_freq'] == 0:
                    print(f'GAT training: time elapsed= {(time.time() - time_start):.2f} [s] |'
                          f' epoch={epoch + 1} | batch={batch_idx + 1} | train micro-F1={micro_f1}.')

                # 保存checkpoint
                if config['checkpoint_freq'] is not None and (epoch + 1) % config['checkpoint_freq'] == 0 and batch_idx == 0:
                    ckpt_model_name = f'gat_{config["dataset_name"]}_ckpt_epoch_{epoch + 1}.pth'
                    config['test_perf'] = -1  # 尚未进行性能测试
                    torch.save(utils.get_training_state(config, gat), os.path.join(CHECKPOINTS_PATH, ckpt_model_name))

            elif phase == LoopPhase.VAL:
                if config['enable_tensorboard']:
                    writer.add_scalar('val_loss', loss.item(), global_step)
                    writer.add_scalar('val_micro_f1', micro_f1, global_step)

                if config['console_log_freq'] is not None and batch_idx % config['console_log_freq'] == 0:
                    print(f'GAT validation: time elapsed= {(time.time() - time_start):.2f} [s] |'
                          f' epoch={epoch + 1} | batch={batch_idx + 1} | val micro-F1={micro_f1}')

                # 选择最优参数
                if micro_f1 > BEST_VAL_MICRO_F1 or loss.item() < BEST_VAL_LOSS:
                    BEST_VAL_MICRO_F1 = max(micro_f1, BEST_VAL_MICRO_F1)
                    BEST_VAL_LOSS = min(loss.item(), BEST_VAL_LOSS)
                    PATIENCE_CNT = 0
                else:
                    PATIENCE_CNT += 1

                if PATIENCE_CNT >= patience_period:
                    raise Exception('Stopping the training, the universe has no more patience for this training.')
            else:
                return micro_f1  # 单纯的验证，直接返回f1值

    return main_loop


def train_gat_ppi(config):

    # 记录全局参数，最好的验证F1值，最好的验证损失
    global BEST_VAL_MICRO_F1, BEST_VAL_LOSS

    device = torch.device("cuda" if torch.cuda.is_available() and not config['force_cpu'] else "cpu")

    # Step1 加载数据
    data_loader_train, data_loader_val, data_loader_test = load_graph_data(config, device)

    # Step2 准备模型
    gat = GAT_ppi(
        num_of_layers=config['num_of_layers'],
        num_heads_per_layer=config['num_heads_per_layer'],
        num_features_per_layer=config['num_features_per_layer'],
        add_skip_connection=config['add_skip_connection'],
        bias=config['bias'],
        dropout=config['dropout'],
        log_attention_weights=False
    ).to(device)

    # Step3 准备训练工具
    loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = Adam(gat.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    # 返回主迭代方法，这样提高代码复用率
    main_loop = get_main_loop(
        config=config,
        gat=gat,
        sigmoid_cross_entropy_loss=loss_fn,
        optimizer=optimizer,
        patience_period=config['patience_period'],
        time_start=time.time()
    )

    BEST_VAL_MICRO_F1, BEST_VAL_LOSS, PATIENCE_CNT = [0, 0, 0]  # 重置

    # Step4 开始训练过程
    for epoch in range(config['num_of_epochs']):
        # 训练循环
        main_loop(phase=LoopPhase.TRAIN, data_loader=data_loader_train, epoch=epoch)

        # 验证循环
        with torch.no_grad():
            try:
                main_loop(phase=LoopPhase.VAL, data_loader=data_loader_val, epoch=epoch)
            except Exception as e:
                print(str(e))
                break

    # Step5 验证
    if config['should_test']:
        micro_f1 = main_loop(phase=LoopPhase.TEST, data_loader=data_loader_test)
        config['test_perf'] = micro_f1

        print('*' * 50)
        print(f'Test micro-F1 = {micro_f1}')

    else:
        config['test_perf'] = -1

    # 保存最新的GAT模型的二进制文件
    torch.save(
        utils.get_training_state(config, gat),
        os.path.join(BINARIES_PATH, utils.get_available_binary_name(config['dataset_name']))
    )


if __name__ == '__main__':
    train_gat_ppi(get_training_args())





















