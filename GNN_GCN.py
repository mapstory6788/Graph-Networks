import os
import time
import random
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder

# 配置项
class configs():
    def __init__(self):
        # Data
        self.data_path = r'./data/cora'
        self.save_model_dir = './'

        self.model_name = r'GAT'
        self.seed = 2023

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epoch = 500
        self.in_features = 1433  # core ~ feature:1433
        self.hidden_features = 16  # 隐层数量
        self.output_features = 8  # core~paper-point~ 8类

        self.learning_rate = 0.01
        self.dropout = 0.5

        self.istrain = True
        self.istest = True


cfg = configs()


def seed_everything(seed=2023):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


seed_everything(seed=cfg.seed)


# 读取Cora数据集 return geometric Data格式
def index_to_mask(index, size):
    mask = np.zeros(size, dtype=bool)
    mask[index] = True
    return mask


def load_cora_data(data_path=cfg.data_path):
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

    data.train_mask = index_to_mask(idx_train, size=labels.shape[0])
    data.val_mask = index_to_mask(idx_val, size=labels.shape[0])
    data.test_mask = index_to_mask(idx_test, size=labels.shape[0])

    return data


class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=8, dropout=cfg.dropout, bias=True):
        super(GAT, self).__init__()

        self.conv1 = GATConv(in_channels, out_channels, heads=heads, concat=True, dropout=dropout, bias=bias)
        self.conv2 = GATConv(heads * out_channels, out_channels, heads=heads, concat=False, dropout=dropout, bias=bias)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class myGAT_run():
    def train(self):
        t = time.time()
        dataset = load_cora_data()
        model = GAT(dataset.num_features, cfg.output_features).to(cfg.device)
        data = dataset
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=5e-4)

        model.train()
        for epoch in range(cfg.epoch):
            optimizer.zero_grad()
            output = model(data)
            preds = output.max(dim=1)[1]
            loss_train = F.nll_loss(output[data.train_mask], data.y[data.train_mask].long())
            correct = preds[data.train_mask].eq(data.y[data.train_mask]).sum().item()
            acc_train = correct / int(data.train_mask.sum())
            loss_train.backward()
            optimizer.step()
            loss_val = F.nll_loss(output[data.val_mask], data.y[data.val_mask].long())
            correct = preds[data.val_mask].eq(data.y[data.val_mask]).sum().item()
            acc_val = correct / int(data.val_mask.sum())
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'acc_train: {:.4f}'.format(acc_train),
                  'loss_val: {:.4f}'.format(loss_val.item()),
                  'acc_val: {:.4f}'.format(acc_val),
                  'time: {:.4f}s'.format(time.time() - t))
        torch.save(model, os.path.join(cfg.save_model_dir, 'latest.pth'))  # 模型保存

    def infer(self):
        # Create Test Processing
        dataset = load_cora_data()
        data = dataset
        model_path = os.path.join(cfg.save_model_dir, 'latest.pth')
        model = torch.load(model_path, map_location=torch.device(cfg.device))
        model.eval()
        output = model(data)
        params = sum(p.numel() for p in model.parameters())
        preds = output.max(dim=1)[1]
        loss_test = F.nll_loss(output[data.test_mask], data.y[data.test_mask].long())
        correct = preds[data.test_mask].eq(data.y[data.test_mask]).sum().item()
        acc_test = correct / int(data.test_mask.sum())
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test),
              'params={:.4f}k'.format(params / 1024))


if __name__ == '__main__':
    mygraph = myGAT_run()
    if cfg.istrain == True:
        mygraph.train()
    if cfg.istest == True:
        mygraph.infer()