import torch
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
import networkx as nx
import igraph as ig

from utils.constants import DatasetType, GraphVisualizationTool


def plot_in_out_degree_distributions(edge_index, num_of_nodes, dataset_name):
    """
    绘制入度和出度的分布
    :param edge_index: 边集
    :param num_of_nodes: 节点数量
    :param dataset_name: 数据集名
    """
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()

    assert isinstance(edge_index, np.ndarray), f'Expected NumPy array got {type(edge_index)}.'

    my_font = font_manager.FontProperties(fname="/usr/share/fonts/truetype/arphic/uming.ttc")

    # 保存每个节点的入度和出度(二者应该相等在Cora 和 PPI 这样的无向图数据集中)
    in_degrees = np.zeros(num_of_nodes, dtype=np.int)
    out_degrees = np.zeros(num_of_nodes, dtype=np.int)

    # Edge index Shape = (2,E)，第一行为出节点，第二行为入节点
    num_of_nodes = edge_index.shape[1]
    for cnt in range(num_of_nodes):
        source_node_id = edge_index[0, cnt]
        target_node_id = edge_index[1, cnt]

        out_degrees[source_node_id] += 1
        in_degrees[target_node_id] += 1

    hist = np.zeros(np.max(out_degrees) + 1)  # 由于入度和出度一样，保存每个度数的数量

    for out_degree in out_degrees:
        hist[out_degree] += 1

    fig = plt.figure(figsize=(12, 8), dpi=100)  # dpi是分辨率，即每英寸多少个像素，缺省值为80
    fig.subplots_adjust(hspace=0.6)

    plt.subplot(311)
    plt.plot(in_degrees, color='red')
    plt.xlabel('node_id')
    plt.ylabel('入度', fontproperties=my_font)
    plt.title('不同节点的入度图', fontproperties=my_font)

    plt.subplot(312)
    plt.plot(out_degrees, color='green')
    plt.xlabel('node_id')
    plt.ylabel('出度', fontproperties=my_font)
    plt.title('不同节点的出度图', fontproperties=my_font)

    plt.subplot(313)
    plt.plot(hist, color='blue')
    plt.xlabel('node_degree')
    plt.ylabel('节点度的数量', fontproperties=my_font)
    plt.title('节点度的分布图', fontproperties=my_font)
    plt.xticks(np.arange(0, len(hist), 20.0))

    plt.grid(True)  # 显示网格
    plt.show()


def visulization_graph(edge_index, node_labels, dataset_name, visualization_tool=GraphVisualizationTool.IGRAPH):

    # if isinstance(edge_index, torch.Tensor):
    edge_index_np = edge_index

    # isinstance(node_labels, torch.Tensor):
    node_labels_np = node_labels

    num_of_nodes = len(node_labels_np)
    edge_index_tuples = list(zip(edge_index_np[0, :], edge_index_np[1, :]))  # igraph requires this format

    if visualization_tool == GraphVisualizationTool.NETWORKX:
        nx_graph = nx.Graph()
        nx_graph.add_edges_from(edge_index_tuples)
        nx.draw_networkx(nx_graph)
        plt.show()

    elif visualization_tool == GraphVisualizationTool.IGRAPH:
        # 构建igraph graph
        ig_graph = ig.Graph()
        ig_graph.add_vertices(num_of_nodes)
        ig_graph.add_edges(edge_index_tuples)

        # 编写可视化设置config
        visual_style = {}

        visual_style["bbox"] = (3000, 3000)
        visual_style["margin"] = 35

        # 定义每条边的权重，边的显示厚度正比与权重
        edge_weights_raw = np.clip(np.log(np.asarray(ig_graph.edge_betweenness()) + 1e-16), a_min=0, a_max=None)
        edge_weights_raw_normalized = edge_weights_raw / np.max(edge_weights_raw)  # 归一化
        edge_weights = [w ** 6 for w in edge_weights_raw_normalized]  # 此处的6可以随意改
        visual_style["edge_width"] = edge_weights

        # 定义顶点大小，顶点大小正比与顶点的度大小
        visual_style["vertex_size"] = [deg / 2 for deg in ig_graph.degree()]

        # 设置2D布局，采用kamada_kawai()来强化仿真效果
        visual_style['layout'] = ig_graph.layout_kamada_kawai()

        print('Plotting results ... (it may take couple of seconds).')
        ig.plot(ig_graph, **visual_style)


def draw_entropy_histogram(entropy_array, title, color='blue', uniform_distribution=False, num_bins=30):
    max_value = np.max(entropy_array)
    bar_width = (max_value / num_bins) * (1.0 if uniform_distribution else 0.75)
    histogram_values, histogram_bins = np.histogram(entropy_array, bins=num_bins, range=(0.0, max_value))
    """ histogram(a,bins=10,range=None,weights=None,density=False)
        a 是统计数据的数组
        bins 指定统计的区间个数
        range 是一个长度为2的元组，表示统计范围的最小值和最大值，默认值None，表示范围由数据的范围决定
        weights 为数组的每个元素指定了权值,histogram()会对区间中数组所对应的权值进行求和
        density 为True时，返回每个区间的概率密度；为False，返回每个区间中元素的个数

        histogram_values 为落在各个bins的数值个数
        histogram_bins 各个bins
    """

    plt.bar(histogram_bins[: num_bins], histogram_values[:num_bins], width=bar_width, color=color)
    plt.xlabel(f'entropy bins')
    plt.ylabel(f'# of node neighborhoods')
    plt.title(title)