import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F


############################################ 对比学习 ############################################
def cross_entropy_max_distance(z, temperature=0.3):
    """
    使用标准对比学习方法最大化样本间距离

    参数:
        z: 样本特征，形状 [B, N, D]
        temperature: 温度参数
    返回:
        loss: 对比损失
    """
    if len(z.shape) == 3:
        B, N, D = z.shape
        if N <= 1:
            return torch.tensor(0.0, device=z.device)
        # 特征归一化
        z = F.normalize(z, p=2, dim=2)
        # 计算相似度矩阵
        sim_matrix = torch.matmul(z, z.transpose(1, 2)) / temperature  # [B, N, N]
        # 对每个样本，构建其正样本和负样本
        # 在这个场景中，假设每个样本的正样本是它自己（相似度为1），其他都是负样本
        # 计算exp相似度
        exp_sim = torch.exp(sim_matrix)  # [B, N, N]
        # 对角线元素（自身相似度）
        diag_exp = torch.exp(torch.ones(B, N, device=z.device) / temperature)  # [B, N]
        # 每行的负样本exp相似度之和
        neg_exp_sum = torch.sum(exp_sim * (1 - torch.eye(N, device=z.device).unsqueeze(0)), dim=2)  # [B, N]
        # NT-Xent损失
        loss = -torch.log(diag_exp / (diag_exp + neg_exp_sum) + 1e-8)  # [B, N]
    else:
        N, D = z.shape
        if N <= 1:
            return torch.tensor(0.0, device=z.device)
        # 特征归一化
        z = F.normalize(z, p=2, dim=1)
        # 计算相似度矩阵
        sim_matrix = torch.matmul(z, z.transpose(0, 1)) / temperature  # [N, N]
        # 对每个样本，构建其正样本和负样本
        # 在这个场景中，假设每个样本的正样本是它自己（相似度为1），其他都是负样本
        # 计算exp相似度
        exp_sim = torch.exp(sim_matrix)  # [N, N]
        # 对角线元素（自身相似度）
        diag_exp = torch.exp(torch.ones(N, device=z.device) / temperature)  # [ N]
        # 每行的负样本exp相似度之和
        neg_exp_sum = torch.sum(exp_sim * (1 - torch.eye(N, device=z.device)), dim=1)  # [N]
        # NT-Xent损失
        loss = -torch.log(diag_exp / (diag_exp + neg_exp_sum) + 1e-8)  # [N]
    return torch.mean(loss)


def cluster_center_anchor_info_nce(
        route_node,  # [B, R, D]
        route_node_info,  # [B, R, K, D]
        temperature=0.1
):
    """
    InfoNCE loss where:
        - anchors = cluster centers (route_node)
        - positives = nodes within the same cluster (route_node_info[b, r])
        - negatives = nodes from all other clusters

    Returns:
        scalar loss (mean over B * R anchors)
    """
    B, R, D = route_node.shape
    K = route_node_info.shape[2]

    # Normalize for cosine similarity
    route_node = F.normalize(route_node, dim=-1)  # [B, R, D]
    route_node_info = F.normalize(route_node_info, dim=-1)  # [B, R, K, D]

    # Flatten all non-center nodes: [B, R*K, D]
    all_nodes = route_node_info.view(B, R * K, D)  # [B, R*K, D]

    # Compute similarity between each center and all nodes
    # route_node: [B, R, D], all_nodes: [B, R*K, D]
    # Use batch matrix multiplication: [B, R, D] × [B, D, R*K] → [B, R, R*K]
    logits = torch.bmm(route_node, all_nodes.transpose(1, 2))  # [B, R, R*K]
    logits = logits / temperature

    logits_grouped = logits.view(B, R, R, K)  # [B, R (anchor), R (source cluster), K]

    # For anchor r, positives are from cluster r → logits_grouped[:, r, r, :]
    pos_logits = logits_grouped.diagonal(dim1=1, dim2=2)  # [B, K, R] → transpose 取对角线
    pos_logits = pos_logits.transpose(1, 2)  # [B, R, K]

    # Numerator: log(sum(exp(pos_logits)))
    # Denominator: log(sum(exp(all_logits))) = logsumexp over last dim
    log_prob_pos = torch.logsumexp(pos_logits, dim=-1)  # [B, R]
    log_prob_all = torch.logsumexp(logits, dim=-1)  # [B, R]

    # InfoNCE loss per anchor
    loss_per_anchor = -(log_prob_pos - log_prob_all)  # [B, R]

    # Mean over all anchors (B * R)
    loss = loss_per_anchor.mean()

    return loss


############################################ 对embedding之后的序列数据进行二次embedding（实际未起作用） ############################################
class PatchEmbed(nn.Module):
    def __init__(self, dim, patch_len, stride=None, pos=True):
        super().__init__()
        self.patch_len = patch_len
        self.stride = patch_len if stride is None else stride
        self.patch_proj = nn.Conv1d(dim, dim, patch_len, stride=stride, padding=0)

    def forward(self, x):
        # x:  (B*C,T,D)--> (B*C,D,T)--> (B*C,D,N)--> (B*C,N,D)
        x = self.patch_proj(x.transpose(1, 2)).transpose(1, 2)
        return x


class MultiScalePatchEmbed(nn.Module):
    def __init__(self, dim, patch_lens, stride=None, pos=True):
        super().__init__()
        self.patch_lens = patch_lens
        self.dim = dim
        self.pos = pos

        # 为每个patch_len创建对应的PatchEmbed模块
        self.patch_embeds = nn.ModuleList()
        for patch_len in patch_lens:
            stride_val = patch_len if stride is None else stride
            if patch_len == 1:
                patch_embed = nn.Identity()
            else:
                patch_embed = PatchEmbed(dim, patch_len, stride_val, pos)
            self.patch_embeds.append(patch_embed)

    def forward(self, x):  # (B*C,T,D)
        embeddings = []
        for patch_embed in self.patch_embeds:
            emb = patch_embed(x)
            embeddings.append(emb)
        return embeddings


############################################ 多尺度时序特征提取+节点路由增强（多尺度实际未起作用） ############################################
class Tem_Extract_Layer(nn.Module):
    def __init__(self, seq_len, model_dim, patch_list, num_nodes, dropout, num_heads, feed_forward_dim):
        super(Tem_Extract_Layer, self).__init__()
        self.seq_len = seq_len
        self.model_dim = model_dim
        self.patch_list = patch_list
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.num_heads = num_heads
        self.feed_forward_dim = feed_forward_dim
        assert self.patch_list[0] == 1, "patch_list[0] must be 1"
        self.patch_list = self.patch_list
        self.num_patches_list = []
        for patch_len in self.patch_list:
            self.num_patches_list.append(seq_len // patch_len)

        self.temp_list = nn.ModuleList()
        for i in range(len(self.patch_list)):
            self.temp_list.append(
                SelfAttentionLayer(self.model_dim, self.feed_forward_dim, self.num_heads, self.dropout, mask=True))

    def forward(self, x, dim=None):  # [b,t,c,d] (BC,T+1,D)
        # res = x
        x = x.permute(0, 2, 1, 3).reshape(-1, self.seq_len + 1, self.model_dim)
        node_info = x[:, -1, :]
        x = x[:, :-1, :]
        D = x.shape[-1]
        C = self.num_nodes
        B = x.shape[0] // C

        x_list = [x]
        node_info = node_info.unsqueeze(1).reshape(B * C, 1, D)  # [B*C, 1, D]

        multi_patch_out = []  # 对不同粒度分别进行特征提取，其中node_info在不同层间共享
        for i in range(len(x_list))[::-1]:  # 从粗到细进行特征处理
            mask = create_temporal_marker_mask(x_list[i].shape[1] + 1).to(x.device)  # [T, T]
            out = self.temp_list[i](torch.cat([x_list[i], node_info], dim=1), dim=1, mask=mask)  # [B*C, N+1, D]
            node_info = out[:, -1:, :]
            multi_patch_out.append(out[:, :-1, :])

        y = multi_patch_out[0]
        y = torch.cat([y, node_info], 1).reshape(-1, self.num_nodes, self.seq_len + 1, self.model_dim).permute(0, 2, 1, 3)
        return y  # (BC,T,D)


############################################ MASK FOR TRANSFORMER ############################################
def create_temporal_marker_mask(seq_len):
    """
    创建一个特殊的注意力掩码，使得:
    - 前seq_len-1个时间步只能相互交互
    - 最后一个时间步可以访问所有时间步

    参数:
        seq_len: 总序列长度T

    返回:
        mask: 形状为[T, T]的注意力掩码
              mask[i, j] = 1 表示时间步i可以关注时间步j
    """
    # 创建全零掩码矩阵
    mask = torch.zeros(seq_len, seq_len)

    # 前T-1个时间步可以相互交互（排除最后一个标记）
    mask[:seq_len - 1, :seq_len - 1] = 1

    # 最后一个标记可以访问所有时间步
    mask[seq_len - 1, :] = 1

    return mask


############################################ SELF-ATTENTION ############################################
class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value, mask=None):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
                             query @ key
                     ) / self.head_dim ** 0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            assert mask is not None, "Please provide a mask."
            attn_score.masked_fill_(mask == 0, -np.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out


class SelfAttentionLayer(nn.Module):
    def __init__(
            self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()
        self.mask = mask
        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2, mask=None):
        if self.mask:
            assert mask is not None, "Please provide a mask."
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        out = self.attn(x, x, x, mask)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


############################################ 空间信息扩散 ############################################
class Spa_Propagate_Layer(nn.Module):
    def __init__(self, seq_len, model_dim, patch_list, num_nodes, dropout, num_heads, feed_forward_dim):
        super(Spa_Propagate_Layer, self).__init__()
        self.seq_len = seq_len
        self.model_dim = model_dim
        self.patch_list = patch_list
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.num_heads = num_heads
        self.feed_forward_dim = feed_forward_dim
        assert self.patch_list[0] == 1, "patch_list[0] must be 1"
        self.patch_list = self.patch_list
        self.num_patches_list = []
        for patch_len in self.patch_list:
            self.num_patches_list.append(seq_len // patch_len)

        self.temp_list = nn.ModuleList()
        for i in range(len(self.patch_list)):
            self.temp_list.append(
                SelfAttentionLayer(self.model_dim, self.feed_forward_dim, self.num_heads, self.dropout, mask=False)
            )

    def forward(self, x, dim=None):  # [b,t,c,d] (BC,T+1,D)
        # res = x
        x = x.permute(0, 2, 1, 3).reshape(-1, self.seq_len + 1, self.model_dim)
        node_info = x[:, -1, :]
        x = x[:, :-1, :]
        D = x.shape[-1]
        C = self.num_nodes
        B = x.shape[0] // C

        x_list = [x]
        node_info = node_info.unsqueeze(1).reshape(B * C, 1, D)  # [B*C, 1, D]

        multi_patch_out = []  # 对不同粒度分别进行特征提取，其中node_info在不同层间共享
        for i in range(len(x_list))[::-1]:  # 从粗到细进行特征处理
            out = self.temp_list[i](torch.cat([x_list[i], node_info], dim=1), dim=1)  # [B*C, N+1, D]
            multi_patch_out.append(out[:, :-1, :])

        y = multi_patch_out[0]
        y = torch.cat([y, node_info], 1).reshape(-1, self.num_nodes, self.seq_len + 1, self.model_dim).permute(0, 2, 1, 3)
        return y  # (BC,T,D)



############################################ 多尺度空间信息扩散（多尺度实际未起作用） ############################################


class Node_Routing_SelfAttentionLayer(nn.Module):
    def __init__(self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0):
        super().__init__()
        self.head = num_heads
        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, route, mode='atten'):  # (B,R,K,D) (B,R,K,D)
        # x: (batch_size, ..., length, model_dim)
        residual = route
        B, R, K, _ = route.shape
        H = self.head
        D = self.head_dim
        if mode == 'atten':
            # 数据初步分成两类：1.时间步数据 2.路由数据
            query = self.FC_Q(route).reshape(B, R, K, H, -1)
            key = self.FC_K(route).reshape(B, R, K, H, -1)
            value = self.FC_V(route).reshape(B, R, K, H, D)

            # k组T*K的邻接矩阵，表示每个点位中时间步和每个点位路由间的相似度（避免了时间步不对齐的问题
            atten_score = torch.einsum('br khd, br nhd -> br knh', query, key)
            atten_score = torch.softmax(atten_score / key.shape[-1] ** 0.5, dim=3)  #
            out = torch.einsum('br knh, br nhd -> br khd', atten_score, value).reshape(B, R, K, H * D)

            out = self.out_proj(out)

            out = self.dropout1(out)
            out = self.ln1(residual + out)

            residual = out
            out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
            out = self.dropout2(out)
            out = self.ln2(residual + out)
        else:  # 处理未被选中的节点路由
            value = self.FC_V(route).reshape(B, R, K, H, D)
            out = value.reshape(B, R, K, H * D)

            out = self.out_proj(out)

            # out = self.attn(route, route, x)  # (batch_size, ..., length, model_dim)
            out = self.dropout1(out)
            out = self.ln1(residual + out)

            residual = out
            out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
            out = self.dropout2(out)
            out = self.ln2(residual + out)

        return out


class Spa_Extract_Layer(nn.Module):
    def __init__(self, model_dim, cluster_num, input_embedding_dim, space_embedding_dim, dropout, feed_forward_dim, head, topk):
        super(Spa_Extract_Layer, self).__init__()
        self.model_dim = model_dim
        self.route_num = cluster_num
        self.d_model = input_embedding_dim
        self.s_dim = space_embedding_dim
        self.dropout = dropout
        self.feed_forward_dim = feed_forward_dim
        self.head = head
        self.topk = topk

        self.routing_center = nn.init.xavier_uniform_(
            nn.Parameter(torch.empty(self.route_num, self.d_model), requires_grad=True))  # (R,D)
        self.routing_spa = nn.init.xavier_uniform_(nn.Parameter(torch.empty(self.route_num, self.s_dim)))

        self.node_routing_fusion = nn.Linear(model_dim, model_dim)

        self.attention_space = Node_Routing_SelfAttentionLayer(self.model_dim, self.feed_forward_dim, self.head, self.dropout)
        self.attention_space_MLP = Node_Routing_SelfAttentionLayer(self.model_dim, self.feed_forward_dim, self.head, self.dropout)

    def forward(self, node_routing, tem_routing):
        # 计算路由和节点信息间的相似度
        # 结合时间信息进行路由中心初始化
        B, T, C, D = node_routing.shape
        R = self.route_num
        K = self.topk
        node_routing_center = torch.cat([self.routing_center.unsqueeze(0).repeat(B, 1, 1),
                                         tem_routing.unsqueeze(1).repeat(1, self.routing_center.size(0), 1),
                                         self.routing_spa.unsqueeze(0).repeat(B, 1, 1)], -1)

        # 计算路由和各个node_info间的相似度
        node_routing_center = self.node_routing_fusion(node_routing_center)
        route_similarity = torch.softmax(torch.einsum('bcd,brd->brc', node_routing.squeeze(1), node_routing_center), dim=2)  # [B,R,C]
        loss_center = cross_entropy_max_distance(node_routing_center)

        # 根据相似度，对节点分类，找出每个路由对应的topk个节点
        topk_similarities, topk_idx = torch.topk(route_similarity, self.topk, dim=2)  # [B,R,topk]
        bs_idx = torch.arange(B, device=node_routing.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, T, R, K)  # (B,T,R,K)
        ts_idx = torch.arange(node_routing.size(1), device=node_routing.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(B, -1, R, K)  # (B,T,R,K)
        route_node_routing = node_routing[bs_idx, ts_idx, topk_idx.unsqueeze(1).expand(-1, T, -1, -1)]  # (B,T,R,K,D)
        route_node_routing = route_node_routing.permute(0, 2, 3, 1, 4)  # (B,R,K,T,D)

        # 对node_routing进行更新
        route_node_routing = route_node_routing.squeeze(3)
        route_node_routing_update = self.attention_space(route_node_routing, mode='atten')
        route_node_routing_update = route_node_routing_update.unsqueeze(3)

        # 还原route_node_info_updated到node_info中
        #     创建一个新的张量来累积更新值和计数
        updated_node_routing_per_route = torch.zeros([B, R, C, T, D], device=node_routing.device)  # [B,R,C,T,D]
        count = torch.zeros([B, C], device=node_routing.device)  # [B, C]
        similarity_sum_per_route = torch.zeros([B, R, C], device=node_routing.device)  # [B, R, C]

        topk_idx_expanded = topk_idx.unsqueeze(3).unsqueeze(4)  # 形状：(B,R,K,1,1)
        topk_idx_expanded = topk_idx_expanded.repeat(1, 1, 1, T, D)  # 扩展为：(B,R,K,T,D)

        # 按索引填充值（inplace操作）
        updated_node_routing_per_route = updated_node_routing_per_route.scatter(
            dim=2,  # 在第2维(C维)上进行scatter
            index=topk_idx_expanded,  # [B,R,K,T,D]
            src=route_node_routing_update  # [B,R,K,T,D]
        )
        similarity_sum_per_route = similarity_sum_per_route.scatter(
            dim=2,  # 在第2维(C维)上进行scatter
            index=topk_idx,  # [B,R,K,1] -> 扩展以匹配维度
            src=topk_similarities  # [B,R,K,1]
        )
        # 统计每个节点的使用次数
        flat_topk_idx = topk_idx.view(B, -1)  # [B, R*K，1]
        count.scatter_add_(1, flat_topk_idx,
                           torch.ones([B, R * K], device=node_routing.device, dtype=count.dtype))  # [B, C, T]

        # 计算每个路由r在每个变量C上的占比
        total_similarity_per_batch_node = similarity_sum_per_route.sum(dim=1, keepdim=True).repeat(1, R, 1)  # [B, R, C]
        # 创建有效节点掩码
        valid_nodes = total_similarity_per_batch_node > 0  # [B, R, C]
        # 初始化占比张量
        route_proportion = torch.zeros_like(similarity_sum_per_route)  # [B, R, C]
        # 只对有效节点计算占比
        route_proportion[valid_nodes] = (
                similarity_sum_per_route[valid_nodes] /
                total_similarity_per_batch_node[valid_nodes]
        )
        # 对无效节点，按照1/route_num进行赋值(部分节点被选中，但是注意力分数可能嫉妒接近0，对此类节点特殊处理,提升前期鲁棒性)
        invalid_nodes = ~valid_nodes
        cc = count[:, :].unsqueeze(1).repeat(1, R, 1)[invalid_nodes]
        route_proportion[invalid_nodes] = 1.0 / torch.where(cc == 0, torch.ones_like(cc), cc)
        # 扩展route_proportion以匹配updated_node_routing_per_route的维度
        route_proportion_expanded = route_proportion.unsqueeze(-1).unsqueeze(-1)  # [B, R, C, 1, 1]
        route_proportion_expanded = route_proportion_expanded.expand(-1, -1, -1, T, D)  # [B, R, C, T, D]

        # 应用权重
        weighted_updated_node_routing = updated_node_routing_per_route * route_proportion_expanded  # [B, R, C, T, D]

        # 如果需要聚合所有路由的加权贡献，可以沿着路由维度求和
        updated_node_routing = weighted_updated_node_routing.sum(dim=1)  # [B, C, T, D]

        # 保留未被选中的节点的原始信息
        selected_mask = (count > 0).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        unselect_node_info = self.attention_space_MLP(node_routing, mode='mlp').permute(0, 2, 1, 3)
        final_node_routing = torch.where(selected_mask, updated_node_routing, unselect_node_info)

        final_node_routing = final_node_routing.permute(0, 2, 1, 3)
        loss_cluster = cluster_center_anchor_info_nce(node_routing_center, route_node_routing_update.squeeze(3))
        return final_node_routing, topk_idx, loss_center + loss_cluster
