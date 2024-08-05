import torch.nn as nn
from torch_geometric.nn import RGCNConv, GraphConv, GATConv, TransformerConv, GCN2Conv, GAT


class GCN(nn.Module):
    def __init__(self, g_dim, h1_dim, h2_dim, args):
        super(GCN, self).__init__()
        self.num_relations = 2 * args.n_speakers ** 2
        self.conv1 = RGCNConv(g_dim, h1_dim, self.num_relations, num_bases=10)
        self.conv2 = GraphConv(h1_dim, h2_dim)

    def forward(self, node_features, edge_index, edge_type):
        x = self.conv1(node_features, edge_index, edge_type)
        x = self.conv2(x, edge_index)
        return x


class GAT(nn.Module):
    def __init__(self, g_dim, h1_dim, h2_dim, heads, args):
        super(GAT, self).__init__()
        self.num_relations = 2 * args.n_speakers ** 2
        self.conv1 = RGCNConv(g_dim, h1_dim, self.num_relations, num_bases=20)
        self.conv2 = GATConv(h1_dim, h2_dim, heads, dropout=0.6)

    def forward(self, node_features, edge_index, edge_type, return_attention_weights=False):
        x = self.conv1(node_features, edge_index, edge_type)
        x, attn_weights = self.conv2(x, edge_index, return_attention_weights=True)
        if return_attention_weights:
            return x, attn_weights
        return x


class SGCN(nn.Module):
    def __init__(self, g_dim, h1_dim, h2_dim, args):
        self.num_relations = 4
        super(SGCN, self).__init__()
        self.conv1 = TransformerConv(g_dim, h2_dim)

    def forward(self, node_features, edge_index, edge_type):
        x = self.conv1(node_features, edge_index)
        return x
