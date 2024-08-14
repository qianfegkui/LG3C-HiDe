from .ContextualEncoder import ContextualEncoder
from .EdgeAtt import EdgeAtt
from .GCN import GCN, SGCN, GAT
from .Classifier import Classifier
from .functions import batch_graphify, batch_graphifyt, batch_graphifys, batch_graphify_
import himallgg
import torch.nn.functional as F
from .Fusion import *


class CrossModalAttention(nn.Module):
    def __init__(self, d_model):
        super(CrossModalAttention, self).__init__()
        self.d_model = d_model
        self.q_layer = nn.Linear(d_model, d_model)
        self.k_layer = nn.Linear(d_model, d_model)
        self.v_layer = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, y, mask=None):
        '''
        x: [batch_size, seq_len_x, d_model]
        y: [batch_size, seq_len_y, d_model]
        mask: [batch_size, seq_len_x, seq_len_y]
        '''
        q = self.q_layer(x)  # [batch_size, seq_len_x, d_model]
        k = self.k_layer(y)  # [batch_size, seq_len_y, d_model]
        v = self.v_layer(y)  # [batch_size, seq_len_y, d_model]

        # calculate attention scores
        attn_scores = torch.bmm(q, k.transpose(1, 2))  # [batch_size, seq_len_x, seq_len_y]

        if mask is not None:
            attn_scores.masked_fill_(mask == 0, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch_size, seq_len_x, seq_len_y]

        # apply attention weights to values
        attn_output = torch.bmm(attn_weights, v)  # [batch_size, seq_len_x, d_model]

        # apply dropout and residual connection
        attn_output = self.dropout(attn_output)
        output = attn_output + x  # [batch_size, seq_len_x, d_model]

        return output

log = himallgg.utils.get_logger()

class LGGCN(nn.Module):
    def __init__(self, args):
        super(LGGCN, self).__init__()
        uT_dim = 1024
        uA_dim = 1582
        uV_dim = 342
        g_dim = 320
        h1_dim = 100
        h2_dim = 100
        hc_dim = 100
        tag_size = 6
        n_head = 8
        self.wp = args.wp
        self.wf = args.wf
        self.device = args.device
        self.rnn = ContextualEncoder(uT_dim, g_dim, args)
        self.rnn_A = ContextualEncoder(uA_dim, g_dim, args)
        self.rnn_V = ContextualEncoder(uV_dim, g_dim, args)
        self.cross_attention = CrossModalAttention(g_dim)
        self.edge_att = EdgeAtt(g_dim, args)
        self.gat = GAT(g_dim*3, h1_dim, h2_dim, n_head, args)
        self.gcn = GCN(g_dim*3, h1_dim, h2_dim, args)

        self.clf = Classifier(h2_dim*n_head+g_dim*3, hc_dim, tag_size, args)

        self.clf_T = Classifier(g_dim, hc_dim, tag_size, args)
        self.clf_A = Classifier(g_dim, hc_dim, tag_size, args)
        self.clf_V = Classifier(g_dim, hc_dim, tag_size, args)
        self.clf_FM = Classifier(g_dim, g_dim, 2, args)

        self.gcn1 = SGCN(g_dim * 2, h1_dim, g_dim, args)
        self.gcn2 = SGCN(g_dim * 2, h1_dim, g_dim, args)
        self.gcn3 = SGCN(g_dim * 2, h1_dim, g_dim, args)
        self.att = MultiHeadedAttention(10, g_dim)
        self.args = args

        edge_type_to_idx = {}
        for j in range(args.n_speakers):
            for k in range(args.n_speakers):
                edge_type_to_idx[str(j) + str(k) + '0'] = len(edge_type_to_idx)
                edge_type_to_idx[str(j) + str(k) + '1'] = len(edge_type_to_idx)
        self.edge_type_to_idx = edge_type_to_idx
        edge_type_to_idxs = {}
        for j in range(args.n_speakers):
            for k in range(args.n_speakers):
                edge_type_to_idxs[str(j) + str(k)] = len(edge_type_to_idxs)
        self.edge_type_to_idxs = edge_type_to_idxs
        self.edge_type_to_idxt = {'0': 0, '1': 1}
        self.edge_type_to_idx_ = {'0': 0}

    def get_rep(self, data):
        node_features_T = self.rnn(data["text_len_tensor"].cpu(), data["text_tensor"])  # [batch_size, mx_len, D_g]
        node_features_A = self.rnn_A(data["text_len_tensor"].cpu(), data["audio_tensor"])  # [batch_size, mx_len, D_g]
        node_features_V = self.rnn_V(data["text_len_tensor"].cpu(), data["visual_tensor"])  # [batch_size, mx_len, D_g]

        node_features = torch.cat((node_features_T, node_features_A, node_features_V), 2)

        features_T, edge_index_T, edge_type_T, edge_index_lengths_T = batch_graphifys(
            node_features_T, data["text_len_tensor"], data["speaker_tensor"], self.wp, self.wf,
            self.edge_type_to_idxs, self.device)

        features_A, edge_index_A, edge_type_A, edge_index_lengths_A = batch_graphifys(
            node_features_A, data["text_len_tensor"], data["speaker_tensor"], self.wp, self.wf,
            self.edge_type_to_idxs, self.device)

        features_V, edge_index_V, edge_type_V, edge_index_lengths = batch_graphifys(
            node_features_V, data["text_len_tensor"], data["speaker_tensor"], self.wp, self.wf,
            self.edge_type_to_idxs, self.device)

        features, edge_index, edge_type, edge_index_lengths = batch_graphify(
            node_features, data["text_len_tensor"], data["speaker_tensor"], self.wp, self.wf,
            self.edge_type_to_idx, self.device)

        class_output_T = self.clf_FM.get_fea(features_T, data["text_len_tensor"])
        class_output_A = self.clf_FM.get_fea(features_A, data["text_len_tensor"])
        class_output_V = self.clf_FM.get_fea(features_V, data["text_len_tensor"])

        features_T = torch.cat([features_T, class_output_T], dim=-1)
        features_A = torch.cat([features_A, class_output_A], dim=-1)
        features_V = torch.cat([features_V, class_output_V], dim=-1)

        graph_out_T = self.gcn1(features_T, edge_index_T, edge_type_T)
        graph_out_A = self.gcn2(features_A, edge_index_A, edge_type_A)
        graph_out_V = self.gcn3(features_V, edge_index_V, edge_type_V)

        fea1 = self.att(graph_out_T, graph_out_A, graph_out_A)
        fea2 = self.att(graph_out_T, graph_out_V, graph_out_V)
        features_graph = torch.cat([graph_out_T, fea1.squeeze(1), fea2.squeeze(1)], dim=-1)

        graph_out, attn_weights = self.gat(features_graph, edge_index, edge_type, return_attention_weights=True)

        return graph_out, features, graph_out_T, graph_out_A, graph_out_V, class_output_T, class_output_A, class_output_V

    def forward(self, data):
        graph_out, features, graph_out_T, graph_out_A, graph_out_V, class_output_T, class_output_A, class_output_V = self.get_rep(data)
        score = self.clf.get_prob1(torch.cat([features, graph_out], dim=-1), data["text_len_tensor"])
        score_T = self.clf_T.get_prob1(graph_out_T, data["text_len_tensor"])
        score_A = self.clf_A.get_prob1(graph_out_A, data["text_len_tensor"])
        score_V = self.clf_V.get_prob1(graph_out_V, data["text_len_tensor"])
        scores = score + 0.5 * score_T + 0.1 * score_A + 0.1 * score_V
        log_prob = F.log_softmax(scores, dim=-1)
        y_hat = torch.argmax(log_prob, dim=-1)
        return y_hat

    def get_loss(self, data):
        graph_out, features, graph_out_T, graph_out_A, graph_out_V, class_output_T, class_output_A, class_output_V = self.get_rep(data)

        loss = self.clf.get_loss(torch.cat([features, graph_out], dim=-1),
                                 data["label_tensor"], data["text_len_tensor"])
        loss_T = self.clf_T.get_loss(graph_out_T, data["label_tensor"], data["text_len_tensor"])
        loss_A = self.clf_A.get_loss(graph_out_A, data["label_tensor"], data["text_len_tensor"])
        loss_V = self.clf_V.get_loss(graph_out_V,data["label_tensor"], data["text_len_tensor"])

        loss_gender_T = self.clf_FM.get_loss(class_output_T, data["xingbie_tensor"], data["text_len_tensor"])
        loss_gender_A = self.clf_FM.get_loss(class_output_A, data["xingbie_tensor"], data["text_len_tensor"])
        loss_gender_V = self.clf_FM.get_loss(class_output_V, data["xingbie_tensor"], data["text_len_tensor"])

        total_loss = loss + 0.5 * loss_T + 0.1 * loss_A + 0.1 * loss_V+ 0.3 * loss_gender_T + 0.3 * loss_gender_V + 0.3 * loss_gender_A

        return total_loss
