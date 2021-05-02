import time
import os
from collections import defaultdict
import enum

import torch
import matplotlib.pyplot as plt
from scipy.stats import entropy
import numpy as np
import igraph as ig
from sklearn.manifold import TSNE

from utils.utils import *
from utils.data_loading import *
from utils.constants import *
from utils.visualizations import *
from models import GAT_ppi


def visualize_graph_dataset(dataset_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        'dataset_name': dataset_name,
        'should_visualize': True,
        'batch_size': 2,
        'ppi_load_test_only': True
    }
    load_graph_data(config, device)


def visualize_gat_properties(model_name=r'gat_PPI_000000.pth', dataset_name=DatasetType.PPI.name, visualization_type=VisualizationType.ATTENTION):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU, I hope so!
    device = torch.device("cpu")
    config = {
        'dataset_name': dataset_name,
        'should_visualize': False,  # don't visualize the dataset
        'batch_size': 2,  # used only for PPI
        'ppi_load_test_only': True  # used only for PPI (optimization, we're loading only test graphs)
    }

    # Step1：准备数据
    data_loader_test = load_graph_data(config, device)
    node_features, node_labels, topology = next(iter(data_loader_test))
    node_features = node_features.to(device)
    node_labels = node_labels.to(device)
    topology = topology.to(device)

    # Step2：准备模型
    model_path = os.path.join('models/binaries/', model_name)
    model_state = torch.load(model_path)

    gat = GAT_ppi(
        num_of_layers=model_state['num_of_layers'],
        num_heads_per_layer=model_state['num_heads_per_layer'],
        num_features_per_layer=model_state['num_features_per_layer'],
        add_skip_connection=model_state['add_skip_connection'],
        bias=model_state['bias'],
        dropout=model_state['dropout'],
        log_attention_weights=True
    ).to(device)

    print_model_metadata(model_state)

    assert model_state['dataset_name'].lower() == dataset_name.lower(), \
        f"The model was trained on {model_state['dataset_name']} but you're calling it on {dataset_name}."

    gat.load_state_dict(model_state["state_dict"], strict=True)
    gat.eval()

    with torch.no_grad():
        all_nodes_unnormalized_scores, _ = gat((node_features, topology))  # shape = (N, num of classes)
        all_nodes_unnormalized_scores = all_nodes_unnormalized_scores.cpu().numpy()

    if visualization_type == VisualizationType.ATTENTION:
        num_nodes_of_interest = 4  # 选多少个要看的节点
        head_to_visualize = 0  # 要看的head，随便改
        gat_layer_id = 0  # 要看的层

        # assert gat_layer_id == 0, f'Attention visualization for {dataset_name} is only available for the first layer.'

        # 建立完整图
        total_num_of_nodes = len(all_nodes_unnormalized_scores)
        complete_graph = ig.Graph()
        complete_graph.add_vertices(total_num_of_nodes)
        edge_index_tuples = list(zip(topology[0, :], topology[1, :]))
        complete_graph.add_edges(edge_index_tuples)

        nodes_of_interest_ids = np.argpartition(complete_graph.degree(), -num_nodes_of_interest)[-num_nodes_of_interest:]
        """np.argpartition这个函数是快排的partition操作, kth指定第k大的数字，返回的是下标值，在这里就是找度数最大的num_nodes_of_interest个值的下标"""
        random_node_ids = np.random.randint(low=0, high=total_num_of_nodes, size=num_nodes_of_interest)  # 随机取
        nodes_of_interest_ids = np.append(nodes_of_interest_ids, random_node_ids)
        np.random.shuffle(nodes_of_interest_ids)

        target_node_ids = topology[1]
        source_nodes = topology[0]

        for target_node_id in nodes_of_interest_ids:

            # step1 找邻居
            src_nodes_indices = torch.eq(target_node_ids, target_node_id)
            source_node_ids = source_nodes[src_nodes_indices].cpu().numpy()
            size_of_neighborhood = len(source_node_ids)

            # step2 获得标签
            labels = node_labels[source_node_ids].cpu().numpy()

            # step3 获得aij
            all_attention_weights = gat.gat_net[gat_layer_id].attention_weights.squeeze(dim=-1)
            attention_weights = all_attention_weights[src_nodes_indices, head_to_visualize].cpu().numpy()
            attention_weights /= np.max(attention_weights)  # 归一化

            # step4 把原邻居id映射到邻居子图上
            id_to_igraph_id = dict(zip(source_node_ids, range(len(source_node_ids))))

            # step5 建立子图
            ig_graph = ig.Graph()
            ig_graph.add_vertices(size_of_neighborhood)
            ig_graph.add_edges(
                [(id_to_igraph_id[neighbor], id_to_igraph_id[target_node_id]) for neighbor in source_node_ids])

            visual_style = {
                "edge_width": attention_weights,
                "layout": ig_graph.layout_kamada_kawai()
                # layout_kamada_kawai()  仿真强化
                # layout_reingold_tilford_circular()  树状图
            }
            ig.plot(ig_graph, **visual_style)
            image = ig.plot(ig_graph, **visual_style)
            image.save(f'可视化/ppi/node{target_node_id}_layer{gat_layer_id }_head{head_to_visualize}.png')

    elif visualization_type == VisualizationType.ENTROPY:
        num_heads_per_layer = [layer.num_of_heads for layer in gat.gat_net]
        num_layers = len(num_heads_per_layer)
        num_of_nodes = len(node_features)

        target_node_ids = topology[1].cpu().numpy()

        for layer_id in range(num_layers):

            all_attention_weights = gat.gat_net[layer_id].attention_weights.squeeze(dim=-1).cpu().numpy()

            if dataset_name == DatasetType.PPI.name and layer_id < 2:
                print(f'Entropy histograms for {dataset_name} are available only for the first，second layer.')
                break

            for head_id in range(num_heads_per_layer[layer_id]):
                uniform_dist_entropy_list = []
                neighborhood_entropy_list = []

                for target_node_id in range(num_of_nodes):
                    neigborhood_attention = all_attention_weights[target_node_ids == target_node_id].flatten()

                    ideal_uniform_attention = np.ones(len(neigborhood_attention)) / len(neigborhood_attention)

                    neighborhood_entropy_list.append(entropy(neigborhood_attention, base=2))
                    uniform_dist_entropy_list.append(entropy(ideal_uniform_attention, base=2))

                title = f'{dataset_name} entropy histogram layer={layer_id}, attention head={head_id}'

                draw_entropy_histogram(uniform_dist_entropy_list, title, color='orange', uniform_distribution=True)
                draw_entropy_histogram(neighborhood_entropy_list, title, color='dodgerblue')

                fig = plt.gcf()  # get current figure
                plt.show()
                fig.savefig(f'可视化/ppi/layer_{layer_id}_head_{head_id}.jpg')
                plt.close()


class PLAYGROUND(enum.Enum):
    VISUALIZE_DATASET = 0,
    VISUALIZE_GAT = 1


if __name__ == '__main__':
    playground_fn = PLAYGROUND.VISUALIZE_GAT
    if playground_fn == PLAYGROUND.VISUALIZE_DATASET:
        visualize_graph_dataset(dataset_name=DatasetType.PPI.name)

    elif playground_fn == PLAYGROUND.VISUALIZE_GAT:
        visualize_gat_properties(
            model_name=r'gat_PPI_000000.pth',
            dataset_name=DatasetType.PPI.name,
            visualization_type=VisualizationType.ENTROPY
        )
    else:
        raise Exception(f'Woah, this playground function "{playground_fn}" does not exist.')








