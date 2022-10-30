from torch.utils.data.dataset import Dataset
import torch
import numpy as np


class DKTDataSet(Dataset):
    def __init__(self, init_data, data, similar_records, device):
        self.init_data = init_data
        self.data = data
        self.similar_records = similar_records
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        similar_records = self.similar_records[index]
        topK_sub_seq, len_sub = self.top_k_sub_seq(similar_records)
        myself = torch.from_numpy(data)
        others = torch.from_numpy(topK_sub_seq).long()
        other_len = torch.from_numpy(len_sub)
        return myself.to(self.device), others.to(self.device), other_len.to(self.device)

    def top_k_sub_seq(self, similar_records):
        top_k = 18
        len_sub_seq = np.zeros([50, top_k])
        one_student = np.zeros([50, top_k, 50, 3])
        for idx, K in enumerate(similar_records):
            sub_seqs = np.zeros([top_k, 50, 3])
            for idy, i in enumerate(K):
                if idy == top_k :
                    break
                len_sub_seq[idx][idy] = i[1]
                sub_seqs[idy, :i[1], :] = self.init_data[i[0], :i[1]]
            one_student[idx] = sub_seqs
        return one_student, len_sub_seq
