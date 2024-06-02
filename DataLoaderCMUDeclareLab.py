import os
import pickle
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

from Config import Data_path_DecLab


def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
        
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# construct a word2id mapping that automatically takes increment when new words are encountered
word2id = defaultdict(lambda: len(word2id))
UNK = word2id['<unk>']
PAD = word2id['<pad>']

# turn off the word2id - define a named function here to allow for pickling
def return_unk():
    return UNK

def get_length(x):
    return x.shape[1]-(np.sum(x, axis=-1) == 0).sum(1)

class MOSI:
    def __init__(self):
        try:
            self.train = load_pickle(Data_path_DecLab + '/mosi_train.pkl')
            self.dev = load_pickle(Data_path_DecLab + '/mosi_valid.pkl')
            self.test = load_pickle(Data_path_DecLab + '/mosi_test.pkl')
            self.pretrained_emb, self.word2id = None, None
        except:
            print('Cannot find dataset files, trying to create new.')
            pickle_filename = os.path.join(Data_path_DecLab, 'mosi_data_noalign.pkl')
            csv_filename = os.path.join(Data_path_DecLab, 'MOSI-label.csv')

            with open(pickle_filename,'rb') as f:
                d = pickle.load(f)
            
            # read csv file for label and text
            df = pd.read_csv(csv_filename)
            text = df['text']
            vid = df['video_id']
            cid = df['clip_id']
            
            train_split_noalign = d['train']
            dev_split_noalign = d['valid']
            test_split_noalign = d['test']

            # a sentinel epsilon for safe division, without it we will replace illegal values with a constant
            EPS = 1e-6

            # place holders for the final train/dev/test dataset
            self.train = train = []
            self.dev = dev = []
            self.test = test = []
            self.word2id = word2id

            # define a regular expression to extract the video ID out of the keys
            pattern = re.compile('(.*)_(.*)')
            num_drop = 0 # a counter to count how many data points went into some processing issues

            if True:
                v = np.concatenate((train_split_noalign['vision'],dev_split_noalign['vision'], test_split_noalign['vision']),axis=0)
                vlens = get_length(v)

                a = np.concatenate((train_split_noalign['audio'],dev_split_noalign['audio'], test_split_noalign['audio']),axis=0)
                alens = get_length(a)
                
                label = np.concatenate((train_split_noalign['labels'],dev_split_noalign['labels'], test_split_noalign['labels']),axis=0)

                L_V = v.shape[1]
                L_A = a.shape[1]

            all_id = np.concatenate((train_split_noalign['id'], dev_split_noalign['id'], test_split_noalign['id']),axis=0)[:,0]
            all_id_list = list(map(lambda x: x.decode('utf-8'), all_id.tolist()))

            train_size = len(train_split_noalign['id'])
            dev_size = len(dev_split_noalign['id'])
            test_size = len(test_split_noalign['id'])

            dev_start = train_size
            test_start = train_size + dev_size

            all_csv_id = [(vid[i], str(cid[i])) for i in range(len(vid))]

            for i, idd in enumerate(all_id_list):
                # get the video ID and the features out of the aligned dataset
                idd1, idd2 = re.search(pattern, idd).group(1,2)

                # matching process
                try:
                    index = all_csv_id.index((idd1,idd2))
                except:
                    exit()
                """
                    Retrive noalign data from pickle file 
                """
                _words = text[index].split()
                _label = label[i].astype(np.float32)
                _visual = v[i]
                _acoustic = a[i]
                _vlen = vlens[i]
                _alen = alens[i]
                _id = all_id[i]

                # remove nan values
                _visual = np.nan_to_num(_visual)
                _acoustic = np.nan_to_num(_acoustic)

                # remove speech pause tokens - this is in general helpful
                # we should remove speech pauses and corresponding visual/acoustic features together
                # otherwise modalities would no longer be aligned
                actual_words = []
                words = []
                visual = []
                acoustic = []

                # For non-align setting
                # we also need to record sequence lengths
                """TODO: Add length counting for other datasets 
                """
                for word in _words:
                    actual_words.append(word)

                visual = _visual[L_V - _vlen:,:]
                acoustic = _acoustic[L_A - _alen:,:]

                # z-normalization per instance and remove nan/infs
                # visual = np.nan_to_num((visual - visual.mean(0, keepdims=True)) / (EPS + np.std(visual, axis=0, keepdims=True)))
                # acoustic = np.nan_to_num((acoustic - acoustic.mean(0, keepdims=True)) / (EPS + np.std(acoustic, axis=0, keepdims=True)))
                if i < dev_start:
                    train.append(((words, visual, acoustic, actual_words, _vlen, _alen), _label, idd))
                elif i >= dev_start and i < test_start:
                    dev.append(((words, visual, acoustic, actual_words, _vlen, _alen), _label, idd))
                elif i >= test_start:
                    test.append(((words, visual, acoustic, actual_words, _vlen, _alen), _label, idd))
                else:
                    print(f"Found video that doesn't belong to any splits: {idd}")

            print(f"Total number of {num_drop} datapoints have been dropped.")
            print("Dataset split")
            print("Train Set: {}".format(len(train)))
            print("Validation Set: {}".format(len(dev)))
            print("Test Set: {}".format(len(test)))
            word2id.default_factory = return_unk

            # Save glove embeddings cache too
            # self.pretrained_emb = pretrained_emb = load_emb(word2id, config.word_emb_path)
            # torch.save((pretrained_emb, word2id), CACHE_PATH)

            # Save pickles
            to_pickle(train, Data_path_DecLab + '/mosi_train.pkl')
            to_pickle(dev, Data_path_DecLab + '/mosi_dev.pkl')
            to_pickle(test, Data_path_DecLab + '/mosi_test.pkl')

    def get_data(self, mode):
        if mode == "train":
            return self.train, self.word2id, None
        elif mode == "valid":
            return self.dev, self.word2id, None
        elif mode == "test":
            return self.test, self.word2id, None
        else:
            print("Mode is not set properly (train/valid/test)")
            exit()

class MOSEI:
    def __init__(self,):
        try:
            self.train = load_pickle(Data_path_DecLab + '/mosei_train.pkl')
            self.dev = load_pickle(Data_path_DecLab + '/mosei_valid.pkl')
            self.test = load_pickle(Data_path_DecLab + '/mosei_test.pkl')
            self.pretrained_emb, self.word2id = None, None

        except:
            
            # first we align to words with averaging, collapse_function receives a list of functions
            # dataset.align(text_field, collapse_functions=[avg])
            # load pickle file for unaligned acoustic and visual source
            pickle_filename = os.path.join(Data_path_DecLab, 'mosei_senti_data_noalign.pkl')
            csv_filename = os.path.join(Data_path_DecLab, 'MOSEI-label.csv')

            with open(pickle_filename, 'rb') as f:
                d = pickle.load(f)
            
            # read csv file for label and text
            df = pd.read_csv(csv_filename)
            text = df['text']
            vid = df['video_id']
            cid = df['clip_id']
            
            train_split_noalign = d['train']
            dev_split_noalign = d['valid']
            test_split_noalign = d['test']

            # a sentinel epsilon for safe division, without it we will replace illegal values with a constant
            EPS = 1e-6

            # place holders for the final train/dev/test dataset
            self.train = train = []
            self.dev = dev = []
            self.test = test = []
            self.word2id = word2id

            # define a regular expression to extract the video ID out of the keys
            # pattern = re.compile('(.*)\[.*\]')
            pattern = re.compile('(.*)_([.*])')
            num_drop = 0 # a counter to count how many data points went into some processing issues

            v = np.concatenate((train_split_noalign['vision'],dev_split_noalign['vision'], test_split_noalign['vision']),axis=0)
            vlens = get_length(v)

            a = np.concatenate((train_split_noalign['audio'],dev_split_noalign['audio'], test_split_noalign['audio']),axis=0)
            alens = get_length(a)
            
            label = np.concatenate((train_split_noalign['labels'],dev_split_noalign['labels'], test_split_noalign['labels']),axis=0)

            L_V = v.shape[1]
            L_A = a.shape[1]


            all_id = np.concatenate((train_split_noalign['id'], dev_split_noalign['id'], test_split_noalign['id']),axis=0)[:,0]
            all_id_list = all_id.tolist()

            train_size = len(train_split_noalign['id'])
            dev_size = len(dev_split_noalign['id'])
            test_size = len(test_split_noalign['id'])

            dev_start = train_size
            test_start = train_size + dev_size

            all_csv_id = [(vid[i], str(cid[i])) for i in range(len(vid))]

            for i, idd in enumerate(all_id_list):
                # get the video ID and the features out of the aligned dataset

                # matching process
                try:
                    index = i
                except:
                    import ipdb; ipdb.set_trace()

                _words = text[index].split()
                _label = label[i].astype(np.float32)
                _visual = v[i]
                _acoustic = a[i]
                _vlen = vlens[i]
                _alen = alens[i]
                _id = '{}[{}]'.format(all_csv_id[0], all_csv_id[1])           

                # remove nan values
                # label = np.nan_to_num(label)
                _visual = np.nan_to_num(_visual)
                _acoustic = np.nan_to_num(_acoustic)

                # remove speech pause tokens - this is in general helpful
                # we should remove speech pauses and corresponding visual/acoustic features together
                # otherwise modalities would no longer be aligned
                actual_words = []
                words = []
                visual = []
                acoustic = []
                
                for word in _words:
                    actual_words.append(word)

                visual = _visual[L_V - _vlen:,:]
                acoustic = _acoustic[L_A - _alen:,:]

                if i < dev_start:
                    train.append(((words, visual, acoustic, actual_words, _vlen, _alen), _label, idd))
                elif i >= dev_start and i < test_start:
                    dev.append(((words, visual, acoustic, actual_words, _vlen, _alen), _label, idd))
                elif i >= test_start:
                    test.append(((words, visual, acoustic, actual_words, _vlen, _alen), _label, idd))
                else:
                    print(f"Found video that doesn't belong to any splits: {idd}")
                

            # print(f"Total number of {num_drop} datapoints have been dropped.")
            print(f"Total number of {num_drop} datapoints have been dropped.")
            print("Dataset split")
            print("Train Set: {}".format(len(train)))
            print("Validation Set: {}".format(len(dev)))
            print("Test Set: {}".format(len(test)))
            word2id.default_factory = return_unk

            # Save glove embeddings cache too
            # self.pretrained_emb = pretrained_emb = load_emb(word2id, config.word_emb_path)
            # torch.save((pretrained_emb, word2id), CACHE_PATH)
            self.pretrained_emb = None

            # Save pickles
            to_pickle(train, Data_path_DecLab + '/mosei_train.pkl')
            to_pickle(dev, Data_path_DecLab + '/mosei_valid.pkl')
            to_pickle(test, Data_path_DecLab + '/mosei_test.pkl')

    def get_data(self, mode):

        if mode == "train":
            return self.train, self.word2id, self.pretrained_emb
        elif mode == "valid":
            return self.dev, self.word2id, self.pretrained_emb
        elif mode == "test":
            return self.test, self.word2id, self.pretrained_emb
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()

# from Train import bert_tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only = True, do_lower_case=True)

class MSADataset(Dataset):
    def __init__(self, dataset_name, mode):
        ## Fetch dataset
        if "mosi" in dataset_name:
            dataset = MOSI()
        elif "mosei" in dataset_name:
            dataset = MOSEI()
        else:
            print("Dataset not defined correctly")
            exit()
        
        self.data, self.word2id, _ = dataset.get_data(mode)
        self.len = len(self.data)

    @property
    def tva_dim(self):
        t_dim = 768
        return t_dim, self.data[0][0][1].shape[1], self.data[0][0][2].shape[1]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


def get_loader(dataset_name, mode, batch_size, time_len=100, shuffle=True, persistent_workers=False, num_workers=4, pin_memory=False, drop_last=False):
    """Load DataLoader of given DialogDataset"""

    dataset = MSADataset(dataset_name, mode)
    
    # if mode == 'train':
    #     dataset_samples =  len(dataset)
    # elif mode == 'valid':
    #     dataset_samples = len(dataset)
    # elif mode == 'test':
    #     dataset_samples = len(dataset)

    def collate_fn(batch):
        '''
        Collate functions assume batch = [Dataset[i] for i in index_set]
        '''
        # for later use we sort the batch in descending order of length
        batch = sorted(batch, key=lambda x: len(x[0][3]), reverse=True)

        v_lens = []
        a_lens = []
        labels = []
        ids = []

        for sample in batch:
            if len(sample[0]) > 4: # unaligned case
                v_lens.append(torch.IntTensor([sample[0][4]]))
                a_lens.append(torch.IntTensor([sample[0][5]]))
            else:   # aligned cases
                v_lens.append(torch.IntTensor([len(sample[0][3])]))
                a_lens.append(torch.IntTensor([len(sample[0][3])]))
            labels.append(torch.from_numpy(sample[1]))
            ids.append(sample[2])
        vlens = torch.cat(v_lens)
        alens = torch.cat(a_lens)
        labels = torch.cat(labels, dim=0)
        
        # MOSEI sentiment labels locate in the first column of sentiment matrix
        if labels.size(1) == 7:
            labels = labels[:,0][:,None]

        # Rewrite this
        def pad_sequence(sequences, target_len=-1, batch_first=False, padding_value=0.0):
            if target_len < 0:
                max_size = sequences[0].size()
                trailing_dims = max_size[1:]
            else:
                max_size = target_len
                trailing_dims = sequences[0].size()[1:]

            max_len = max([s.size(0) for s in sequences])
            if batch_first:
                out_dims = (len(sequences), max_len) + trailing_dims
            else:
                out_dims = (max_len, len(sequences)) + trailing_dims

            out_tensor = sequences[0].new_full(out_dims, padding_value)
            for i, tensor in enumerate(sequences):
                length = tensor.size(0)
                # use index notation to prevent duplicate references to the tensor
                if batch_first:
                    out_tensor[i, :length, ...] = tensor
                else:
                    out_tensor[:length, i, ...] = tensor
            return out_tensor

        sentences = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch],padding_value=PAD)
        visual = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch], target_len=vlens.max().item())
        acoustic = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch],target_len=alens.max().item())

        ## BERT-based features input prep

        # SENT_LEN = min(sentences.size(0),50)
        SENT_LEN = time_len
        # Create bert indices using tokenizer

        bert_details = []
        for sample in batch:
            text = " ".join(sample[0][3])
            encoded_bert_sent = bert_tokenizer.encode_plus(
                text, max_length=SENT_LEN, add_special_tokens=True, truncation=True, padding='max_length')
            bert_details.append(encoded_bert_sent)

        # Bert things are batch_first
        bert_sentences = torch.LongTensor([sample["input_ids"] for sample in bert_details])
        bert_sentence_types = torch.LongTensor([sample["token_type_ids"] for sample in bert_details])
        bert_sentence_att_mask = torch.LongTensor([sample["attention_mask"] for sample in bert_details])

        # lengths are useful later in using RNNs
        lengths = torch.LongTensor([len(sample[0][0]) for sample in batch])
        if (vlens <= 0).sum() > 0:
            vlens[np.where(vlens == 0)] = 1

        return sentences, acoustic.transpose(0, 1), visual.transpose(0, 1), alens, vlens, labels, bert_sentences, bert_sentence_types, bert_sentence_att_mask, lengths, ids

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        persistent_workers=persistent_workers, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last,
        collate_fn=collate_fn)

    return data_loader

if __name__ == '__main__':
    data_loader = get_loader('mosi_dec', mode='train', batch_size=32)
    for i, batch in enumerate(data_loader):
        sentences, acoustic, visual, alens, vlens, labels, bert_sentences, bert_sentence_types, bert_sentence_att_mask, lengths, ids = batch
        print(bert_sentences.shape, visual.shape, acoustic.shape, labels.shape)
        print(sentences)
        # print(vlens, alens, labels, lengths, bert_sentences, sep='\n')
        # input()
