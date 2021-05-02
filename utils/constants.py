import os
import enum
from torch.utils.tensorboard import SummaryWriter


# 支持的数据集，定义枚举
class DatasetType(enum.Enum):
    PPI = 0


# 可视化工具
class GraphVisualizationTool(enum.Enum):
    NETWORKX = 0,
    IGRAPH = 1


# 3 different model training/eval phases used in train.py
class LoopPhase(enum.Enum):
    TRAIN = 0,
    VAL = 1,
    TEST = 2


class VisualizationType(enum.Enum):
    ATTENTION = 0,
    ENTROPY = 1


writer = SummaryWriter()  # (tensorboard) writer will output to ./runs/ directory by default

# 用于提前停止的全局参数
BEST_VAL_MICRO_F1 = 0
BEST_VAL_LOSS = 0
PATIENCE_CNT = 0

# 文件的路径
BINARIES_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'models', 'binaries')  # 模型文件
CHECKPOINTS_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'models', 'checkpoints')  # checkpoint文件
DATA_DIR_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'data')  # 数据集
IMAGE_PATH = os.path.join(os.path.dirname(__file__), os.pardir, '可视化')  # 图文件

# 确保文件夹存在，否则建一个文件夹
os.makedirs(BINARIES_PATH, exist_ok=True)
os.makedirs(CHECKPOINTS_PATH, exist_ok=True)

# PPI 数据集config
PPI_PATH = os.path.join(DATA_DIR_PATH, 'ppi')
PPI_URL = 'https://data.dgl.ai/dataset/ppi.zip'

PPI_NUM_INPUT_FEATURES = 50
PPI_NUM_CLASSES = 121