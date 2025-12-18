import torch
import torch.nn as nn
from torchinfo import summary

from Layers.SRCA_COMPONENT import Spa_Extract_Layer, Tem_Extract_Layer, Spa_Propagate_Layer


class SRCA(nn.Module):
    def __init__(
            self,
            num_nodes,
            in_steps=12,
            out_steps=12,
            steps_per_day=288,
            days_per_week=7,
            input_dim=3,
            output_dim=1,
            input_embedding_dim=24,
            tod_embedding_dim=24,
            dow_embedding_dim=24,
            space_embedding_dim=80,
            feed_forward_dim=256,
            num_heads=4,
            num_layers=3,
            dropout=0.1,
            num_clusters=4,
            topk=70,
            patch_list=[1],
            mode='WEIGHT'
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.days_per_week = days_per_week
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.space_embedding_dim = space_embedding_dim
        self.mode = mode
        self.model_dim = (
                input_embedding_dim
                + tod_embedding_dim
                + space_embedding_dim
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_clusters = num_clusters
        self.feed_forward_dim = feed_forward_dim
        self.topk = topk
        self.patch_list = patch_list
        assert self.tod_embedding_dim == self.dow_embedding_dim, "tod_embedding_dim must equal to dow_embedding_dim"
        assert self.patch_list[0]==1, "patch_list[0] must equal to 1"
        assert len(self.patch_list)==1, "len(self.patch_list) must equal to 1"

        self.input_proj = nn.Conv2d(self.input_dim, self.input_embedding_dim, kernel_size=(1, 1), padding=0,
                                    stride=(1, 1))

        self.spa_embedding_pool = nn.init.xavier_uniform_(
            nn.Parameter(torch.empty(self.in_steps + 1, num_nodes, space_embedding_dim))
        )
        self.tem_embedding_pool = nn.init.xavier_uniform_(
            nn.Parameter(torch.empty(steps_per_day * days_per_week, tod_embedding_dim))
        )
        self.tem_embedding_routing_pool = nn.init.xavier_uniform_(
            nn.Parameter(torch.empty(steps_per_day * days_per_week, tod_embedding_dim))
        )

        # 可学习的节点嵌入向量
        self.node_routing_trans = nn.Sequential(nn.Linear(self.in_steps, self.input_embedding_dim // 2),
                                                nn.ReLU(),
                                                nn.Linear(self.input_embedding_dim // 2, self.input_embedding_dim))

        self.attn_layers_t = nn.ModuleList(
            [
                Tem_Extract_Layer(self.in_steps, self.model_dim, self.patch_list, self.num_nodes,
                                  self.dropout, self.num_heads, self.feed_forward_dim)
                for _ in range(num_layers)
            ]
        )

        self.attn_layers_s = nn.ModuleList(
            [
                Spa_Extract_Layer(self.model_dim, num_clusters, self.input_embedding_dim, self.space_embedding_dim,
                                  self.dropout, feed_forward_dim, num_heads, topk)
                for _ in range(num_layers)
            ]
        )

        self.attn_layers_st = nn.ModuleList(
            [
                Spa_Propagate_Layer(self.in_steps, self.model_dim, self.patch_list, self.num_nodes,
                                    self.dropout, self.num_heads, self.feed_forward_dim)
                for _ in range(num_layers)
            ]
        )
        self.w_cat = nn.Linear(self.num_layers, 1, bias=False)

        self.output_proj_all = nn.Linear(
            (self.in_steps + 1) * self.model_dim, out_steps * output_dim
        )

    def forward(self, x, epoch=None):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        B = x.shape[0]

        tod = x[..., 1]
        dow = x[..., 2]
        x = x[..., : self.input_dim]  # [B,T,C,D]

        # 生成节点路由
        node_routing = self.node_routing_trans(x[..., 0].permute(0, 2, 1))
        tem_routing = self.tem_embedding_routing_pool[(tod[:, -1, 0] * self.steps_per_day + dow[:, -1, 0] * self.steps_per_day).long()]
        spa_routing = self.spa_embedding_pool[-1].unsqueeze(0).repeat(B, 1, 1)
        node_routing = torch.cat([node_routing, tem_routing.unsqueeze(1).repeat(1, self.num_nodes, 1), spa_routing], dim=-1)  # (batch_size, num_nodes, model_dim)
        node_routing = node_routing.unsqueeze(1)  # (batch_size, 1, num_nodes, model_dim)

        # 生成序列嵌入
        x = self.input_proj(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        tem_emb = self.tem_embedding_pool[(tod * self.steps_per_day + dow * self.steps_per_day).long()]
        adp_emb = self.spa_embedding_pool[:-1].expand(size=(B, *self.spa_embedding_pool[:-1].shape))
        features = [x]
        features.append(tem_emb)
        features.append(adp_emb)
        x = torch.cat([features[0], features[1], features[2]], dim=-1)

        x_list = []
        node_routing_list = []
        loss_contrast_list = []
        for i in range(self.num_layers):
            # 聚合序列信息到node_info中x
            t_out = self.attn_layers_t[i](torch.cat([x, node_routing], dim=1), dim=1)
            x = t_out[:, :-1, :, :]
            node_routing = t_out[:, -1:, :, :]
            # 空间交互
            node_routing, _, loss_contrast = self.attn_layers_s[i](node_routing, tem_routing)
            # 空间扩散
            st_out = self.attn_layers_st[i](torch.cat([x, node_routing], dim=1), dim=1)
            x = st_out[:, :-1, :, :]

            x_list.append(x)
            node_routing_list.append(node_routing)
            loss_contrast_list.append(loss_contrast)

        loss_contrast = torch.mean(torch.stack(loss_contrast_list, dim=-1), dim=-1)

        x = torch.cat([self.w_cat(torch.stack(x_list, dim=-1)).squeeze(-1),
                       self.w_cat(torch.stack(node_routing_list, dim=-1)).squeeze(-1)], dim=1)

        out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
        out = out.reshape(
            B, self.num_nodes, (self.in_steps + 1) * self.model_dim
        )
        out = self.output_proj_all(out).view(
            B, self.num_nodes, self.out_steps, self.output_dim
        )

        out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)

        return out, loss_contrast


if __name__ == "__main__":
    model = SRCA(207, 12, 12)
    summary(model, [64, 12, 207, 3])
