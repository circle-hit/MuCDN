from tkinter import E
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as math

class LayerNorm(nn.Module):
    def __init__(self, config, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(2*config.hidden_dim))
        self.beta = nn.Parameter(torch.zeros(2*config.hidden_dim))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class MaskedSelfAttention(nn.Module):
    def __init__(self, config):
        super(MaskedSelfAttention, self).__init__()
        if config.hidden_dim % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_dim, config.num_attention_heads))
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_dim / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_dim, self.all_head_size)
        self.key = nn.Linear(config.hidden_dim, self.all_head_size)
        self.value = nn.Linear(config.hidden_dim, self.all_head_size)

        self.dropout = nn.Dropout(0.1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, dis_mask, relative_distance, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        if dis_mask is not None:
            dis_mask = dis_mask.unsqueeze(1).expand_as(attention_scores)
            relative_distance = relative_distance.unsqueeze(1).expand_as(attention_scores)
            if self.config.pos:
                attention_probs = nn.Softmax(dim=-1)(attention_scores.masked_fill(dis_mask==0, -1e9) + relative_distance)
            else:
                attention_probs = nn.Softmax(dim=-1)(attention_scores.masked_fill(dis_mask==0, -1e9))
        else:
            attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, torch.mean(attention_scores, dim=1)

class Imp_Graph_Interact(nn.Module):
    def __init__(self, opt):
        super(Imp_Graph_Interact, self).__init__()
        self.mmha = MaskedSelfAttention(opt)
        self.opt = opt
        self.fc1 = nn.Linear(4*opt.hidden_dim, 4*opt.hidden_dim)
        self.fc2 = nn.Linear(4*opt.hidden_dim, 4*opt.hidden_dim)
        self.fc3 = nn.Linear(8*opt.hidden_dim, 1)
        if opt.pos:
            self.position_embedding = nn.Embedding(25, opt.hidden_dim, padding_idx=0)
            self.position_embedding.weight.data.uniform_(-0.1, 0.1)
            self.trans = nn.Linear(opt.hidden_dim, 1)

    def forward(self, hidden_states, inter_mask, intra_mask, intra_relative_distance, inter_relative_distance, attention_mask=None):
        if self.opt.pos:
            inter_pos = F.relu(self.trans(self.position_embedding(inter_relative_distance.long())).squeeze())
            intra_pos = F.relu(self.trans(self.position_embedding(intra_relative_distance.long())).squeeze())

            G_s, inter_score = self.mmha(hidden_states, inter_mask, inter_pos)
            G_o, intra_score = self.mmha(hidden_states, intra_mask, intra_pos)
        else:
            G_s, inter_score = self.mmha(hidden_states, inter_mask, inter_relative_distance)
            G_o, intra_score = self.mmha(hidden_states, intra_mask, intra_relative_distance)

        total_score = inter_score * inter_mask + intra_score * intra_mask
        mask = inter_mask + intra_mask
        probs = nn.Softmax(dim=-1)(total_score.masked_fill(mask==0, 1e-9))

        E1 = F.relu(self.fc1(torch.cat([hidden_states, G_s, (hidden_states-G_s), (hidden_states*G_s)], dim=-1)))
        E2 = F.relu(self.fc2(torch.cat([hidden_states, G_o, (hidden_states-G_o), (hidden_states*G_o)], dim=-1)))
        gate = torch.sigmoid(self.fc3(torch.cat([E1, E2], dim=-1)))
        out = gate*G_s + (1-gate)*G_o
        return out, probs

class MuCDN(nn.Module):
    def __init__(self, opt, D_m, D_h, n_classes=7):

        super(MuCDN, self).__init__()

        if opt.norm:
            norm_train = True
            self.norm1a = nn.LayerNorm(D_m, elementwise_affine=norm_train)

        self.dropout = nn.Dropout(opt.dropout)

        self.opt = opt
        self.hidden_dim = opt.hidden_dim
        self.num_attention_heads = opt.num_attention_heads
        self.attention_head_size = int(opt.hidden_dim / opt.num_attention_heads)

        self.linear_in = nn.Linear(D_m, D_h)
        self.gru_s = nn.GRUCell(2*opt.hidden_dim, opt.hidden_dim)
        self.gru_o = nn.GRUCell(2*opt.hidden_dim, opt.hidden_dim)
    
        self.imp_graph_mmha = Imp_Graph_Interact(opt)

        layers = [nn.Linear(3*opt.hidden_dim, opt.hidden_dim), nn.ReLU()]
        for _ in range(opt.mlp_layers - 1):
            layers += [nn.Linear(opt.hidden_dim, opt.hidden_dim), nn.ReLU()]
        layers += [self.dropout]
        layers += [nn.Linear(opt.hidden_dim, n_classes)]
        self.smax_fc = nn.Sequential(*layers)

    def forward(self, r1, r2, r3, r4, dis_speaker_mask, dis_adj, inter_adj, intra_adj, intra_relative_distance, inter_relative_distance):
        seq_len, _, feature_dim = r1.size()
        r = (r1 + r2 + r3 + r4) / 4
        if self.opt.norm:
            r = self.norm1a(r.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        r = F.relu(self.linear_in(r))

        H = r.transpose(0, 1)
        imp_out, imp_probs = self.imp_graph_mmha(H, inter_adj, intra_adj, intra_relative_distance, inter_relative_distance)
        
        C_0 = self.gru_s(torch.cat([H[:, 0, :], H[:, 0, :]], dim=-1))
        H_1 = C_0.unsqueeze(1)
        
        for i in range(1, seq_len):
            selected = torch.argmax(dis_adj[:, :, i], dim=1, keepdim=True)
            s_mask = dis_speaker_mask[:, i].unsqueeze(1).float()
            adjs = []
            for item, idx in zip(H_1, selected):
                adj_node = torch.index_select(item, 0, idx)
                adjs.append(adj_node)
            extracted = torch.stack(adjs, dim=0)

            probs = imp_probs[:, i, :i].unsqueeze(1)
            imp_info = torch.matmul(probs, H_1).squeeze()
            
            C_s = self.gru_s(torch.cat([H[:, i, :], imp_info], dim=-1), extracted.squeeze())
            C_o = self.gru_o(torch.cat([H[:, i, :], imp_info], dim=-1), extracted.squeeze())
            
            C = C_s * s_mask + C_o * (1 - s_mask)
            H_temp = C.unsqueeze(1)

            H_1 = torch.cat([H_1, H_temp], dim=1)
        
        final = torch.cat([H, H_1, imp_out], dim=-1)
        
        log_prob = F.log_softmax(self.smax_fc(final.transpose(0, 1)), 2)
        return log_prob