import math

import torch.nn as nn
import torch
import torch.nn.functional as F

class CoSKT(nn.Module):
    def __init__(self, batch_size, num_ques, hidden_dim, device):
        super(CoSKT, self).__init__()
        self.lstm = nn.LSTM(2 * hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.sig = nn.Sigmoid()
        self.emb_q = nn.Embedding(num_ques + 1, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_ques)
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_question = num_ques
        self.bs = batch_size
        self.seq_len = 50
        self.w = nn.Linear(2*hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.project = nn.Sequential(nn.Linear(hidden_dim, hidden_dim, bias=False), nn.BatchNorm1d(hidden_dim),
                                     nn.SELU(inplace=True), nn.Linear(hidden_dim, 110, bias=False), nn.BatchNorm1d(110))
        self.register_buffer("temperature", torch.tensor(0.4))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 49 * 2, batch_size * 49 * 2, dtype=bool)).float())
    def forward(self, data):
        me = data[0].squeeze(-1)  # [B, L, 3]
        Q = self.emb_q(me[:, :, 0])
        A = me[:, :, 2].unsqueeze(-1)
        resp_t = A.repeat(1 , 1, self.hidden_dim).float()
        re_resp_t = (1 - A).repeat(1, 1, self.hidden_dim).float()
        Z = torch.cat([Q.mul(resp_t),
                            Q.mul(re_resp_t)], dim=-1)
        intra_out, _ = self.lstm(Z)  # [B, L, H]
        len_index = data[2].view(-1).long()  # [B, L, k]
        nighs = data[1]  # [B, L, k, L, 3]
        nighs = nighs.view(-1, 50, 3)  # [B*L*k, L, 3]
        ques = nighs[:, :, 0]  # [B*L*k, L]
        ans = nighs[:, :, 2].unsqueeze(-1)  # [B*L*k, L, 1]
        n_x = self.emb_q(ques)  # [B*L*k, L, D]
        resp_t = ans.repeat(1, 1, self.hidden_dim ).float()
        re_resp_t = (1 - ans).repeat(1, 1, self.hidden_dim ).float()
        Z_n = torch.cat([n_x.mul(resp_t),
                       n_x.mul(re_resp_t)], dim=-1)
        nighs_out, _ = self.lstm(Z_n)  # [B*L*k, L, H]
        index = torch.tensor([i for i in range(nighs_out.shape[0])], device=self.device)
        out_final = nighs_out[index, len_index-1, :]  # [B*L*k, H]
        r_ans = Z_n[index, len_index, :] # [B*L*k, H]
        r_ans = r_ans.view(self.bs, 50, 18, -1)
        inter_out = out_final.view(self.bs, 50, 18, -1)
        inter_out = inter_out[:, 1:, :]
        intra_out = intra_out[:, :-1, :]
        inter_out1 = inter_out.mean(2)
        score = self.softmax(intra_out.unsqueeze(2).matmul(inter_out.transpose(-2, -1))/math.sqrt(80))
        inter_out = score.matmul(inter_out).squeeze(2)
        r_ans = score.matmul(r_ans).squeeze(2)
        r_ans = self.w(r_ans)
        loss = self.c_loss(intra_out.contiguous().view(-1, self.hidden_dim), inter_out1.contiguous().view(-1, self.hidden_dim), self.bs*49)
        logits_pre = self.sig(self.fc(intra_out + inter_out + r_ans))
        return logits_pre , loss

    def c_loss(self, emb_i, emb_j, batch_size):
        """
                emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
                z_i, z_j as per SimCLR paper
                """
        z_i = F.normalize(self.project(emb_i), dim=1)
        z_j = F.normalize(self.project(emb_j), dim=1)
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * batch_size)
        return loss
