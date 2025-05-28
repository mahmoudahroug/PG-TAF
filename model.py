import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.graph_conv_block import Graph_Conv_Block
from layers.conv1 import ConvBlock
from layers.seq2seq import Seq2Seq


def calculate_laplacian_with_self_loop(matrix):
    # matrix: (N, V, V)
    matrix = matrix + torch.eye(matrix.shape[1]).to(matrix.device)
    row_sum = matrix.sum(-1)
    d_inv_sqrt = torch.pow(row_sum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = [torch.diag(d_inv_sqrt[i]) for i in range(d_inv_sqrt.shape[0])]
    d_mat_inv_sqrt = torch.stack(d_mat_inv_sqrt, dim=0)
    normalized_laplacian = torch.matmul(d_mat_inv_sqrt, matrix).matmul(d_mat_inv_sqrt)
    return normalized_laplacian


class Model(nn.Module):
    def __init__(self, in_channels, num_types=5, type_embed_dim=8, edge_importance_weighting=False, dropout=0.5):
        super(Model, self).__init__()

        

        self.conv_stage1 = nn.Sequential(
            ConvBlock(in_channels, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        ) # (N, 4, T, V) -> (N, 32, T, V)

        self.type_embedding = nn.Embedding(num_types, type_embed_dim) # (N, V) -> (N, V, E)

        self.conv_stage2 = nn.Sequential(
            ConvBlock(40, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        ) # (N, 32+8, T, V) -> (N, 64, T, V)

        self.conv1d = nn.Sequential(
            ConvBlock(in_channel=in_channels, out_channel=56),
            nn.BatchNorm2d(in_channels),
        ) # (N, 4, T, V) -> (N, 64, T, V)

        self.gcn_temporal_networks = nn.ModuleList([
            Graph_Conv_Block(64, 64, dropout=dropout, residual=True),
            Graph_Conv_Block(64, 64, dropout=dropout, residual=True),
            Graph_Conv_Block(64, 64, dropout=dropout, residual=True),
        ])

        # initialize parameters for edge importance weighting
        # changed torch.ones to torch.randn
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList(
                [nn.Parameter(torch.ones((120, 120))) for i in self.gcn_temporal_networks]
                )
        else:
            self.edge_importance = [1] * len(self.gcn_temporal_networks)

        self.num_node = 120
        self.out_dim_per_node = out_dim_per_node = 2 #(x, y) coordinate
        self.seq2seq_car = Seq2Seq(input_size=(64), hidden_size=out_dim_per_node, num_layers=2, dropout=0.5, isCuda=True)
        self.seq2seq_human = Seq2Seq(input_size=(64), hidden_size=out_dim_per_node, num_layers=2, dropout=0.5, isCuda=True)
        self.seq2seq_bike = Seq2Seq(input_size=(64), hidden_size=out_dim_per_node, num_layers=2, dropout=0.5, isCuda=True)

    def reshape_for_lstm(self, feature):
        # prepare for skeleton prediction model
        '''
        N: batch_size
        C: channel
        T: time_step
        V: nodes
        '''
        N, C, T, V = feature.size() 
        now_feat = feature.permute(0, 3, 2, 1).contiguous() # to (N, V, T, C)
        now_feat = now_feat.view(N*V, T, C) 
        return now_feat

    def reshape_from_lstm(self, predicted):
        # predicted (N*V, T, C)
        NV, T, C = predicted.size()
        now_feat = predicted.view(-1, self.num_node, T, self.out_dim_per_node) # (N, T, V, C) -> (N, C, T, V) [(N, V, T, C)]
        now_feat = now_feat.permute(0, 3, 2, 1).contiguous() # (N, C, T, V)
        return now_feat

    def reshape_to_gcn(self, features):
        # (N, C, T, V) -> (N, T, V, C)
        now_feat = features.permute(0, 2, 3, 1).contiguous()
        return now_feat

    def forward(self, pra_x, pra_A, pra_pred_length, agent_type_ids, pra_teacher_forcing_ratio=0, pra_teacher_location=None):
        x = pra_x

        # normalized_graph = calculate_laplacian_with_self_loop(pra_A)
        # normalized_graph.unsqueeze_(1)

        # out = self.conv1d(x) # (N, 4, T, V) -> (N, 56, T, V)
        out = self.conv_stage1(x) # (N, 4, T, V) -> (N, 32, T, V)
        # Type embeddings
        agent_type_ids = agent_type_ids[:, 0, 0, :]  # (N, V)
        agent_type_ids = agent_type_ids - 1  # shift 1–5 → 0–4
        agent_type_ids = agent_type_ids.clamp(min=0, max=4).long()  # assuming only 5 types: 0,1,2,3,4
        type_embed = self.type_embedding(agent_type_ids)  # (N, V, E)
        type_embed = type_embed.permute(0, 2, 1)          # (N, E, V)
        type_embed = type_embed.unsqueeze(2)              # (N, E, 1, V)
        type_embed = type_embed.expand(-1, -1, x.shape[2], -1)  # (N, E, T, V)

        # Concatenate type embedding to feature map
        out = torch.cat([out, type_embed], dim=1)  # (N, 32 + 8 (E) = 40, T, V)

        out = self.conv_stage2(out)  # (N, 40, T, V) -> (N, 64, T, V)

        normalized_graph = calculate_laplacian_with_self_loop(pra_A).unsqueeze(1)
        
        for net, importance in zip(self.gcn_temporal_networks, self.edge_importance):
            out = self.reshape_to_gcn(out)
            # leaky relu for weak edges
            importance = F.leaky_relu(importance, negative_slope=0.2)
            # normailize importance
            importance = torch.softmax(importance, dim=-1)

            out = net(normalized_graph + importance, out)  # (N, T, V, C + E)

        graph_conv_feature = self.reshape_for_lstm(out)
        last_position = self.reshape_for_lstm(pra_x[:,:2]) #(N, C, T, V)[:, :2] -> (N, T, V*2) [(N*V, T, C)]

        if pra_teacher_forcing_ratio>0 and type(pra_teacher_location) is not type(None):
            pra_teacher_location = self.reshape_for_lstm(pra_teacher_location)

        now_predict_car = self.seq2seq_car(in_data=graph_conv_feature, last_location=last_position[:,-1:,:], pred_length=pra_pred_length, teacher_forcing_ratio=pra_teacher_forcing_ratio, teacher_location=pra_teacher_location)
        now_predict_car = self.reshape_from_lstm(now_predict_car) # (N, C, T, V)

        now_predict_human = self.seq2seq_human(in_data=graph_conv_feature, last_location=last_position[:,-1:,:], pred_length=pra_pred_length, teacher_forcing_ratio=pra_teacher_forcing_ratio, teacher_location=pra_teacher_location)
        now_predict_human = self.reshape_from_lstm(now_predict_human) # (N, C, T, V)

        now_predict_bike = self.seq2seq_bike(in_data=graph_conv_feature, last_location=last_position[:,-1:,:], pred_length=pra_pred_length, teacher_forcing_ratio=pra_teacher_forcing_ratio, teacher_location=pra_teacher_location)
        now_predict_bike = self.reshape_from_lstm(now_predict_bike) # (N, C, T, V)

        now_predict = (now_predict_car + now_predict_human + now_predict_bike)/3.

        return now_predict 
