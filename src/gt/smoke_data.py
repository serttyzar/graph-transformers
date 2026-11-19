"""Малые синтетические данные только для --offline-smoke; не входят в исследование."""
import torch
from torch_geometric.data import Data

from .data import add_encodings, synthetic_graph


def graph_splits(name, lappe_k, rwse_k):
    task = {'zinc': 'regression', 'csl': 'classification', 'peptides-func': 'multilabel'}[name]
    info = dict(node_cats=[1] if name == 'csl' else [28],
                edge_cats=None if name == 'csl' else [4],
                out_dim=1 if name == 'zinc' else 10, task=task,
                metric={'regression': 'mae', 'classification': 'acc', 'multilabel': 'ap'}[task])
    graphs = []
    for i in range(100 if name == 'csl' else 48):
        n = 5 + i % 6
        data = synthetic_graph(n, 4, seed=i)
        generator = torch.Generator().manual_seed(i)
        data.x = torch.randint(0, info['node_cats'][0], (n, 1), generator=generator)
        if info['edge_cats']:
            data.edge_attr = torch.randint(0, 4, (data.edge_index.size(1),), generator=generator)
        if task == 'regression':
            data.y = torch.tensor([i / 48.0])
        elif task == 'classification':
            data.y = torch.tensor([i % 10])
        else:
            data.y = ((torch.arange(10) + i) % 2).float().unsqueeze(0)
        graphs.append(add_encodings(data, lappe_k, rwse_k))
    if name == 'csl':
        return {'all': graphs}, info
    return dict(train=graphs[:24], val=graphs[24:36], test=graphs[36:]), info


def node_data():
    data = synthetic_graph(96, 4, feat_dim=16, seed=0)
    data.y = torch.arange(96) % 3
    split = {'train': torch.arange(48), 'valid': torch.arange(48, 72), 'test': torch.arange(72, 96)}
    return data, split, dict(node_dim=16, out_dim=3, task='classification', metric='acc')
