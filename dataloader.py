import numpy
import torch
from torch.nn.modules import padding
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd

def pad_matrix(matrix, padding_index=0):
    max_len = max(i.size(0) for i in matrix)
    batch_matrix = []
    for item in matrix:
        item = item.numpy()
        batch_matrix.append(numpy.pad(item, ((0, max_len-len(item)), (0, max_len-len(item))), 'constant', constant_values=(padding_index, padding_index)))
    return batch_matrix
    
class MELDDataset(Dataset):

    def __init__(self, split, classify='emotion'):
        '''
        label index mapping = 
        '''
        self.speakers, self.emotion_labels, self.sentiment_labels, \
        self.roberta1, self.roberta2, self.roberta3, self.roberta4, self.discourse_graph, self.discourse_speaker_mask, \
        self.inter_speaker_graph, self.intra_speaker_graph, self.intra_relative_distance, self.inter_relative_distance, \
        self.sentences, self.trainIds, self.testIds, self.validIds, self.multiIds \
        = pickle.load(open('meld/meld_features_roberta_discourse_new.pkl', 'rb'), encoding='latin1')  

        if split == 'train':
            self.keys = [x for x in self.trainIds]
        elif split == 'test':
            self.keys = [x for x in self.testIds]
        elif split == 'valid':
            self.keys = [x for x in self.validIds]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.roberta1[vid]),\
               torch.FloatTensor(self.roberta2[vid]),\
               torch.FloatTensor(self.roberta3[vid]),\
               torch.FloatTensor(self.roberta4[vid]),\
               torch.FloatTensor([1]*len(self.emotion_labels[vid])),\
               torch.LongTensor(self.emotion_labels[vid]),\
               torch.FloatTensor(self.discourse_speaker_mask[vid]),\
               torch.FloatTensor(self.discourse_graph[vid]),\
               torch.FloatTensor(self.inter_speaker_graph[vid]),\
               torch.FloatTensor(self.intra_speaker_graph[vid]),\
               torch.FloatTensor(self.intra_relative_distance[vid]),\
               torch.FloatTensor(self.inter_relative_distance[vid]),\
        
    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        discourse_graph = torch.FloatTensor(pad_matrix(dat[7]))
        inter_speaker_graph = torch.FloatTensor(pad_matrix(dat[8]))
        intra_speaker_graph = torch.FloatTensor(pad_matrix(dat[9]))
        intra_relative_distance = torch.FloatTensor(pad_matrix(dat[10]))
        inter_relative_distance = torch.FloatTensor(pad_matrix(dat[11]))
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<7 else discourse_graph if i<8 else inter_speaker_graph if i<9 else intra_speaker_graph if i<10 else intra_relative_distance if i<11 else inter_relative_distance for i in dat]

class EmoryNLPDataset(Dataset):

    def __init__(self, split):

        '''
        label index mapping =  {'Joyful': 0, 'Mad': 1, 'Peaceful': 2, 'Neutral': 3, 'Sad': 4, 'Powerful': 5, 'Scared': 6}
        '''
        
        self.speakers, self.emotion_labels, \
        self.roberta1, self.roberta2, self.roberta3, self.roberta4, self.discourse_graph, self.discourse_speaker_mask, \
        self.inter_speaker_graph, self.intra_speaker_graph, self.intra_relative_distance, self.inter_relative_distance, \
        self.sentences, self.trainId, self.testId, self.validId, self.multiId \
        = pickle.load(open('emorynlp/emorynlp_features_roberta_discourse.pkl', 'rb'), encoding='latin1')

        self.xIntent, self.xAttr, self.xNeed, self.xWant, self.xEffect, self.xReact, self.oWant, self.oEffect, self.oReact \
        = pickle.load(open('emorynlp/emorynlp_features_comet.pkl', 'rb'), encoding='latin1')
        
        if split == 'train':
            self.keys = [x for x in self.trainId]
        elif split == 'test':
            self.keys = [x for x in self.testId]
        elif split == 'valid':
            self.keys = [x for x in self.validId]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.roberta1[vid]),\
               torch.FloatTensor(self.roberta2[vid]),\
               torch.FloatTensor(self.roberta3[vid]),\
               torch.FloatTensor(self.roberta4[vid]),\
               torch.FloatTensor([1]*len(self.emotion_labels[vid])),\
               torch.LongTensor(self.emotion_labels[vid]),\
               torch.FloatTensor(self.discourse_speaker_mask[vid]),\
               torch.FloatTensor(self.discourse_graph[vid]),\
               torch.FloatTensor(self.inter_speaker_graph[vid]),\
               torch.FloatTensor(self.intra_speaker_graph[vid]),\
               torch.FloatTensor(self.intra_relative_distance[vid]),\
               torch.FloatTensor(self.inter_relative_distance[vid]),\

    def __len__(self):
        return self.len
    
    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        discourse_graph = torch.FloatTensor(pad_matrix(dat[7]))
        inter_speaker_graph = torch.FloatTensor(pad_matrix(dat[8]))
        intra_speaker_graph = torch.FloatTensor(pad_matrix(dat[9]))
        intra_relative_distance = torch.FloatTensor(pad_matrix(dat[10]))
        inter_relative_distance = torch.FloatTensor(pad_matrix(dat[11]))
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<7 else discourse_graph if i<8 else inter_speaker_graph if i<9 else intra_speaker_graph if i<10 else intra_relative_distance if i<11 else inter_relative_distance for i in dat]