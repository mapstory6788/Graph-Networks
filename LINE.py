import os
import pandas as pd
import numpy as np
import networkx as nx
import time
import scipy.sparse as sp
from torch_geometric.data import Data
from torch_geometric.transforms import ToSparseTensor
import torch_geometric.utils
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn


# 配置项
class configs():
    def __init__(self):
        # Data
        self.data_path = r'./data'
        self.save_model_dir = r'./'
        self.num_nodes = 2708
        self.embedding_dim = 128
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.learning_rate = 0.01
        self.epoch = 30
        self.criterion = nn.BCEWithLogitsLoss()

        self.istrain = True
        self.istest = True


cfg = configs()


def load_cora_data(data_path='./data/cora'):
    content_df = pd.read_csv(os.path.join(data_path, "cora.content"), delimiter="\t", header=None)
    content_df.set_index(0, inplace=True)
    index = content_df.index.tolist()
    features = sp.csr_matrix(content_df.values[:, :-1], dtype=np.float32)

    # 处理标签
    labels = content_df.values[:, -1]
    class_encoder = LabelEncoder()
    labels = class_encoder.fit_transform(labels)

    # 读取引用关系
    cites_df = pd.read_csv(os.path.join(data_path, "cora.cites"), delimiter="\t", header=None)
    cites_df[0] = cites_df[0].astype(str)
    cites_df[1] = cites_df[1].astype(str)
    cites = [tuple(x) for x in cites_df.values]
    edges = [(index.index(int(cite[0])), index.index(int(cite[1]))) for cite in cites]
    edges = np.array(edges).T

    # 构造Data对象
    data = Data(x=torch.from_numpy(np.array(features.todense())),
                edge_index=torch.LongTensor(edges),
                y=torch.from_numpy(labels))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    # 读取Cora数据集 return geometric Data格式
    def index_to_mask(index, size):
        mask = np.zeros(size, dtype=bool)
        mask[index] = True
        return mask

    data.train_mask = index_to_mask(idx_train, size=labels.shape[0])
    data.val_mask = index_to_mask(idx_val, size=labels.shape[0])
    data.test_mask = index_to_mask(idx_test, size=labels.shape[0])

    def to_networkx(data):
        edge_index = data.edge_index.to(torch.device('cpu')).numpy()
        G = nx.DiGraph()
        for src, tar in edge_index.T:
            G.add_edge(src, tar)
        return G

    networkx_data = to_networkx(data)

    return data, networkx_data


# 获取数据:pyg_data:torch_geometric格式;networkx_data:networkx格式

def generate_pairs(adj_matrix):
    # 根据邻接矩阵生成正例和负例
    pos_pairs = torch.nonzero(adj_matrix, as_tuple=True)
    pos_u = pos_pairs[0]
    pos_v = pos_pairs[1]

    mask = torch.ones_like(adj_matrix)
    for i in range(len(pos_u)):
        mask[pos_u[i]][pos_v[i]] = 0
        mask[pos_v[i]][pos_u[i]] = 0

    tmp = torch.nonzero(mask, as_tuple=True)

    # TODO 随机选取负例
    idx = torch.randperm(tmp[0].size(0))
    neg_u = tmp[0][idx][:pos_u.size(0)]
    neg_v = tmp[1][idx][:pos_v.size(0)]

    return pos_u, pos_v, neg_u, neg_v


# 构建LINE网络
class LINE(nn.Module):
    def __init__(self, num_nodes, embed_dim):
        super(LINE, self).__init__()
        # num_nodes为Node个数 , embed_dim为描述Node的Embedding维度
        self.embed_dim = embed_dim
        self.num_nodes = num_nodes
        self.embeddings = nn.Embedding(self.num_nodes, self.embed_dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.embeddings.weight.data.normal_(std=1 / self.embed_dim)

    def forward(self, pos_u, pos_v, neg_v):
        emb_pos_u = self.embeddings(pos_u)
        emb_pos_v = self.embeddings(pos_v)
        emb_neg_v = self.embeddings(neg_v)

        pos_scores = torch.sum(torch.mul(emb_pos_u, emb_pos_v), dim=1)
        neg_scores = torch.sum(torch.mul(emb_pos_u, emb_neg_v), dim=1)

        return pos_scores, neg_scores


class LINE_run():
    def train(self):
        t = time.time()
        # 创建一个模型
        _, networkx_data = load_cora_data()
        adj_matrix = torch.tensor(
            nx.adjacency_matrix(networkx_data).toarray()
            , dtype=torch.float32)
        model = LINE(num_nodes=cfg.num_nodes, embed_dim=cfg.embedding_dim).to(cfg.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

        # Train
        model.train()
        for epoch in range(cfg.epoch):
            optimizer.zero_grad()
            pos_u, pos_v, neg_u, neg_v = generate_pairs(adj_matrix)
            pos_u = pos_u.to(cfg.device)
            pos_v = pos_v.to(cfg.device)
            neg_v = neg_v.to(cfg.device)
            pos_scores, neg_scores = model(pos_u, pos_v, neg_v)
            pos_losses = cfg.criterion(pos_scores, torch.ones(len(pos_scores)).to(cfg.device))
            neg_losses = cfg.criterion(neg_scores, torch.zeros(len(neg_scores)).to(cfg.device))
            loss = pos_losses + neg_losses
            loss.backward()
            optimizer.step()
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss.item()),
                  'time: {:.4f}s'.format(time.time() - t))
        torch.save(model, os.path.join(cfg.save_model_dir, 'latest.pth'))  # 模型保存
        print('Embedding dim : ({},{})'.format(model.embeddings.weight.shape[0], model.embeddings.weight.shape[1]))

    def infer(self):
        # Create Test Processing
        _, networkx_data = load_cora_data()
        adj_matrix = torch.tensor(
            nx.adjacency_matrix(networkx_data).toarray()
            , dtype=torch.float32)

        model_path = os.path.join(cfg.save_model_dir, 'latest.pth')
        model = torch.load(model_path, map_location=torch.device(cfg.device))
        model.eval()

        _, networkx_data = load_cora_data()
        pos_u, pos_v, neg_u, neg_v = generate_pairs(adj_matrix)
        pos_u = pos_u.to(cfg.device)
        pos_v = pos_v.to(cfg.device)
        neg_v = neg_v.to(cfg.device)
        pos_scores, neg_scores = model(pos_u, pos_v, neg_v)
        pos_losses = cfg.criterion(pos_scores, torch.ones(len(pos_scores)).to(cfg.device))
        neg_losses = cfg.criterion(neg_scores, torch.zeros(len(neg_scores)).to(cfg.device))
        loss = pos_losses + neg_losses
        print("Test set results:",
              "loss= {:.4f}".format(loss.item()),
              'Embedding dim : ({},{})'.format(model.embeddings.weight.shape[0], model.embeddings.weight.shape[1]))


if __name__ == '__main__':
    mygraph = LINE_run()
    if cfg.istrain == True:
        mygraph.train()
    if cfg.istest == True:
        mygraph.infer()