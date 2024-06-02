import os
import pickle

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from Utils import whether_type_str
from Config import Data_path_local, dataset_scales_mins, dataset_scales_maxs

# AVEC Structure
avec_features = ["text", "mfcc", "ege", "ds", "au", "resnet", 'label']

DATA_PATH = Data_path_local

def multi_collate_avec(batch):
    batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
    
    # get the data out of the batch - use pad sequence util functions from PyTorch to pad things
    labels = torch.Tensor([sample[3] for sample in batch]).reshape(-1,).float()
    if whether_type_str(batch[0][0][0]):
        sentences = [sample[0].tolist() for sample in batch]
    else:
        sentences = pad_sequence([torch.FloatTensor(sample[0]) for sample in batch], padding_value=0).transpose(0, 1)
    acoustic = pad_sequence([torch.FloatTensor(sample[1]) for sample in batch], padding_value=0).transpose(0, 1)
    visual = pad_sequence([torch.FloatTensor(sample[2]) for sample in batch], padding_value=0).transpose(0, 1)
        
    lengths = torch.LongTensor([sample[0].shape[0] for sample in batch])
    return sentences, acoustic, visual, labels, lengths

def get_avec2019_dataset(mode='train', text='text', audio='mfcc', video='au', normalize=[False, False, False], log_scale=[False, False, False],):
    if mode == 'valid':
        mode = 'dev'
    with open(os.path.join(DATA_PATH, 'avec2019', mode+'.pkl'), 'rb') as f:
        data = pickle.load(f)
        
    assert text in avec_features  
    assert audio in avec_features  
    assert video in avec_features
    l_features = [np.nan_to_num(data_[avec_features.index(text)], nan=0.0, posinf=0, neginf=0) for data_ in data]
    a_features = [np.nan_to_num(data_[avec_features.index(audio)], nan=0.0, posinf=0, neginf=0) for data_ in data]
    v_features = [np.nan_to_num(data_[avec_features.index(video)], nan=0.0, posinf=0, neginf=0) for data_ in data]
    labels = [data_[-1] for data_ in data]

    if log_scale[0]:
        l_features = [np.nan_to_num(np.log(f - dataset_scales_mins['avec2019'][0][text] + 1 + 1e-6)) for f in l_features]
    if log_scale[1]:
        a_features = [np.nan_to_num(np.log(f - dataset_scales_mins['avec2019'][1][audio] + 1 + 1e-6)) for f in a_features]
    if log_scale[2]:
        v_features = [np.nan_to_num(np.log(f - dataset_scales_mins['avec2019'][2][video] + 1 + 1e-6)) for f in v_features]

    if normalize[0]:
        max_l, min_l = max([np.max(f) for f in l_features]), min([np.min(f) for f in l_features])
        l_features = [2*(f-min_l)/(max_l-min_l)-1 for f in l_features]
    if normalize[1]:
        max_a, min_a = max([np.max(f) for f in a_features]), min([np.min(f) for f in a_features])
        a_features = [2*(f-min_a)/(max_a-min_a)-1 for f in a_features]
    if normalize[2]:
        max_v, min_v = max([np.max(f) for f in v_features]), min([np.min(f) for f in v_features])
        v_features = [2*(f-min_v)/(max_v-min_v)-1 for f in v_features]
    
    return l_features, a_features, v_features, labels

class AVEC2019Dataset(Dataset):
    def __init__(self, mode, dataset='avec2019', text='text', audio='mfcc', video='au', normalize=[False, False, False], log_scale=[False, False, False]):
        assert mode in ['test', 'train', 'valid']
        assert dataset in ['avec2019']
        self.l_features, self.a_features, self.v_features, self.labels = get_avec2019_dataset(mode=mode, text=text, audio=audio, video=video, normalize=normalize, log_scale=log_scale)

    def __getitem__(self, index):
        return self.l_features[index], self.a_features[index], self.v_features[index], self.labels[index]
    
    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    print('='*80)
    dataset = AVEC2019Dataset(mode='test', dataset='avec2019', text='text', audio='mfcc', video='au', normalize=[False, False, False], log_scale=[False, False, False],)
    data_loader = DataLoader(dataset, 8, collate_fn=multi_collate_avec)
    print('All samples:', len(dataset))
    for i, data in enumerate(data_loader):
        print(len(data[0][0]), [data_.shape for data_ in data[1:]])
        print(data[-1])
