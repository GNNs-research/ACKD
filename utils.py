import numpy as np
import matplotlib.pyplot as plt
import torch
import logging
import pytz
import random
import os
import yaml
import shutil
from datetime import datetime
from ogb.nodeproppred import Evaluator
from dgl import function as fn

CPF_data = ["cora", "citeseer", "pubmed", "a-computer", "a-photo"]
OGB_data = ["ogbn-arxiv", "ogbn-products"]
NonHom_data = ["pokec", "penn94"]
BGNN_data = ["house_class", "vk_class"]

def plot_loss_and_score(file_path):
    # 读取保存的文件
    data = np.load(file_path)
    loss_and_score = data['arr_0']  # 或者使用您保存时的变量名

    # 提取数据
    epochs = loss_and_score[:, 0]
    loss_l = loss_and_score[:, 1]
    loss_val = loss_and_score[:, 2]
    loss_test = loss_and_score[:, 3]
    score_l = loss_and_score[:, 4]
    score_val = loss_and_score[:, 5]
    score_test = loss_and_score[:, 6]

    # 绘制曲线
    plt.figure(figsize=(10, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_l, label='Train Loss')
    plt.plot(epochs, loss_val, label='Validation Loss')
    plt.plot(epochs, loss_test, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    # 绘制得分曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, score_l, label='Train Score')
    plt.plot(epochs, score_val, label='Validation Score')
    plt.plot(epochs, score_test, label='Test Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Score Curve')
    plt.legend()

    plt.tight_layout()
    plt.show()
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# conf = get_training_config(args.exp_setting + args.model_config_path, args.teacher, args.dataset)
def get_training_config(config_path, model_name, dataset):
    with open(config_path, "r") as conf:
        full_config = yaml.load(conf, Loader=yaml.FullLoader)
    dataset_specific_config = full_config["global"]
    model_specific_config = full_config[dataset][model_name]

    if model_specific_config is not None:
        specific_config = dict(dataset_specific_config, **model_specific_config)
    else:
        specific_config = dataset_specific_config

    specific_config["model_name"] = model_name
    return specific_config


def check_writable(path, overwrite=True):
    if not os.path.exists(path):
        os.makedirs(path)
    elif overwrite:
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        pass


def check_readable(path):
    if not os.path.exists(path):
        raise ValueError(f"No such file or directory! {path}")


def timetz(*args):
    tz = pytz.timezone("US/Pacific")
    return datetime.now(tz).timetuple()


def get_logger(filename, console_log=False, log_level=logging.INFO):
    tz = pytz.timezone("US/Pacific")
    log_time = datetime.now(tz).strftime("%b%d_%H_%M_%S")
    logger = logging.getLogger(__name__)
    logger.propagate = False  # avoid duplicate logging
    logger.setLevel(log_level)

    # Clean logger first to avoid duplicated handlers
    for hdlr in logger.handlers[:]:
        logger.removeHandler(hdlr)

    file_handler = logging.FileHandler(filename)
    formatter = logging.Formatter("%(asctime)s: %(message)s", datefmt="%b%d %H-%M-%S")
    formatter.converter = timetz
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console_log:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger


# 这段代码实现了一个函数 idx_split，用于将给定的索引 idx 随机分成两部分，其中一部分包含给定比例 ratio 的元素，另一部分包含剩余的元素。
def idx_split(idx, ratio, seed=0):
    """
    randomly split idx into two portions with ratio% elements and (1 - ratio)% elements
    """
    set_seed(seed)
    n = len(idx)
    cut = int(n * ratio)
    # 在函数内部，首先通过 torch.randperm(n) 生成了 0 到 n-1 的随机排列索引，然后根据指定的比例将这些索引划分为两部分
    idx_idx_shuffle = torch.randperm(n)

    idx1_idx, idx2_idx = idx_idx_shuffle[:cut], idx_idx_shuffle[cut:]
    idx1, idx2 = idx[idx1_idx], idx[idx2_idx]
    # assert((torch.cat([idx1, idx2]).sort()[0] == idx.sort()[0]).all())
    # idx1：包含比例为 ratio 的部分的索引。
    # idx2：包含剩余元素的索引。
    return idx1, idx2


# 这段代码实现了一个函数 graph_split，用于将图数据集按照一定比例进行划分，其中包括训练集、验证集和测试集，以及用于归纳式评估的测试集的进一步划分。
def graph_split(idx_train, idx_val, idx_test, rate, seed):
    """
    Args:
        The original setting was transductive. Full graph is observed, and idx_train takes up a small portion.
        Split the graph by further divide idx_test into [idx_test_tran, idx_test_ind].
        rate = idx_test_ind : idx_test (how much test to hide for the inductive evaluation)
        rate：归纳式评估的比例，即用于归纳式评估的测试节点在总测试节点中所占的比例
        Ex. Ogbn-products
        loaded     : train : val : test = 8 : 2 : 90, rate = 0.2
        after split: train : val : test_tran : test_ind = 8 : 2 : 72 : 18

    Return:
        Indices start with 'obs_' correspond to the node indices within the observed subgraph,
        where as indices start directly with 'idx_' correspond to the node indices in the original graph
    """
    idx_test_ind, idx_test_tran = idx_split(idx_test, rate, seed)

    idx_obs = torch.cat([idx_train, idx_val, idx_test_tran])
    N1, N2 = idx_train.shape[0], idx_val.shape[0]
    obs_idx_all = torch.arange(idx_obs.shape[0])
    obs_idx_train = obs_idx_all[:N1]
    obs_idx_val = obs_idx_all[N1 : N1 + N2]
    obs_idx_test = obs_idx_all[N1 + N2 :]

    return obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind


def get_evaluator(dataset):
    if dataset in CPF_data + NonHom_data + BGNN_data:

        def evaluator(out, labels):
            pred = out.argmax(1)
            return pred.eq(labels).float().mean().item()

    elif dataset in OGB_data:
        ogb_evaluator = Evaluator(dataset)

        def evaluator(out, labels):
            pred = out.argmax(1, keepdim=True)
            input_dict = {"y_true": labels.unsqueeze(1), "y_pred": pred}
            return ogb_evaluator.eval(input_dict)["acc"]

    else:
        raise ValueError("Unknown dataset")

    return evaluator


def get_evaluator(dataset):
    def evaluator(out, labels):
        pred = out.argmax(1)
        return pred.eq(labels).float().mean().item()

    return evaluator


def compute_min_cut_loss(g, out):
    out = out.to("cpu")
    S = out.exp()
    A = g.adj().to_dense()
    D = g.in_degrees().float().diag()
    min_cut = (
        torch.matmul(torch.matmul(S.transpose(1, 0), A), S).trace()
        / torch.matmul(torch.matmul(S.transpose(1, 0), D), S).trace()
    )
    return min_cut.item()


def feature_prop(feats, g, k):
    """
    Augment node feature by propagating the node features within k-hop neighborhood.
    The propagation is done in the SGC fashion, i.e. hop by hop and symmetrically normalized by node degrees.
    """
    assert feats.shape[0] == g.num_nodes()

    degs = g.in_degrees().float().clamp(min=1)
    norm = torch.pow(degs, -0.5).unsqueeze(1)

    # compute (D^-1/2 A D^-1/2)^k X
    for _ in range(k):
        feats = feats * norm
        g.ndata["h"] = feats
        g.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
        feats = g.ndata.pop("h")
        feats = feats * norm

    return feats
