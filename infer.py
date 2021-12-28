# -*- coding: utf-8 -*-

import os
import pickle
import torch
import torch.nn.functional as F
import argparse

from data_utils import ABSADataset, Tokenizer, build_embedding_matrix
from data_utils import ABSADatesetReader
from bucket_iterator import BucketIterator
from models import LSTM, SenticGCN, SenticGCN_BERT
from generate_sentic_dependency_graph import load_sentic_word, dependency_adj_matrix


class Inferer:
    """A simple inference example"""
    def __init__(self, opt):
        self.opt = opt
        fname = {
            'twitter': {
                'train': './datasets/acl-14-short-data/train.raw',
                'test': './datasets/acl-14-short-data/test.raw'
            },
            'rest14': {
                'train': './datasets/semeval14/restaurant_train.raw',
                'test': './datasets/semeval14/restaurant_test.raw'
            },
            'lap14': {
                'train': './datasets/semeval14/laptop_train.raw',
                'test': './datasets/semeval14/laptop_test.raw'
            },
            'rest15': {
                'train': './datasets/semeval15/restaurant_train.raw',
                'test': './datasets/semeval15/restaurant_test.raw'
            },
            'rest16': {
                'train': './datasets/semeval16/restaurant_train.raw',
                'test': './datasets/semeval16/restaurant_test.raw'
            },
        }
        if os.path.exists(opt.dataset+'_word2idx.pkl'):
            print("loading {0} tokenizer...".format(opt.dataset))
            with open(opt.dataset+'_word2idx.pkl', 'rb') as f:
                 word2idx = pickle.load(f)
                 self.tokenizer = Tokenizer(word2idx=word2idx)
        else:
            print("reading {0} dataset...".format(opt.dataset))
            
            text = ABSADatesetReader.__read_text__([fname[opt.dataset]['train'], fname[opt.dataset]['test']])
            self.tokenizer = Tokenizer()
            self.tokenizer.fit_on_text(text)
            with open(opt.dataset+'_word2idx.pkl', 'wb') as f:
                 pickle.dump(self.tokenizer.word2idx, f)
        embedding_matrix = build_embedding_matrix(self.tokenizer.word2idx, opt.embed_dim, opt.dataset)
        self.model = opt.model_class(embedding_matrix, opt).to(opt.device)
        print('loading model {0} ...'.format(opt.model_name))
        self.model.load_state_dict(torch.load(opt.state_dict_path))
        self.model = self.model
        # switch model to evaluation mode
        self.model.eval()
        torch.autograd.set_grad_enabled(False)

    def evaluate(self, raw_text, aspect):
        senticNet = load_sentic_word()
        text_seqs = [self.tokenizer.text_to_sequence(raw_text.lower())]
        aspect_seqs = [self.tokenizer.text_to_sequence(aspect.lower())]
        left_seqs = [self.tokenizer.text_to_sequence(raw_text.lower().split(aspect.lower())[0])]
        text_indices = torch.tensor(text_seqs, dtype=torch.int64)
        aspect_indices = torch.tensor(aspect_seqs, dtype=torch.int64)
        left_indices = torch.tensor(left_seqs, dtype=torch.int64)
        sdat_graph = torch.tensor([dependency_adj_matrix(raw_text.lower(), aspect.lower(), senticNet)])
        data = {
            'text_indices': text_indices, 
            'aspect_indices': aspect_indices,
            'left_indices': left_indices, 
            'sdat_graph': sdat_graph
        }
        t_inputs = [data[col].to(opt.device) for col in self.opt.inputs_cols]
        t_outputs = self.model(t_inputs)

        t_probs = F.softmax(t_outputs, dim=-1).cpu().numpy()
        return t_probs



if __name__ == '__main__':
    dataset = 'rest14'
    # set your trained models here
    model_state_dict_paths = {
        'lstm': 'state_dict/lstm_'+dataset+'.pkl',
        'senticgcn': 'state_dict/senticgcn_'+dataset+'.pkl',
        'senticgcn_bert': 'state_dict/senticgcn_bert_'+dataset+'.pkl',
    }
    model_classes = {
        'lstm': LSTM,
        'senticgcn': SenticGCN,
        'senticgcn_bert': SenticGCN_BERT,
    }
    input_colses = {
        'lstm': ['text_indices'],
        'senticgcn': ['text_indices', 'aspect_indices', 'left_indices', 'sdat_graph'],
        'senticgcn_bert': ['text_bert_indices', 'text_indices', 'aspect_indices', 'bert_segments_indices', 'left_indices', 'sdat_graph'],
    }
    class Option(object): pass
    opt = Option()
    opt.model_name = 'senticgcn'
    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.dataset = dataset
    opt.state_dict_path = model_state_dict_paths[opt.model_name]
    opt.embed_dim = 300
    opt.hidden_dim = 300
    opt.polarities_dim = 3
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    raw_text = 'Food is always fresh and hot - ready to eat !'
    aspect = 'food'

    print('The input are as follows:')
    print('Sentence:', raw_text)
    print('Aspect:', aspect)

    inf = Inferer(opt)
    print('='*10, 'Inferring ......')
    t_probs = inf.evaluate(raw_text, aspect)
    infer_label = t_probs.argmax(axis=-1)[0] - 1
    label_dict = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

    print('The test results is:', infer_label, label_dict[infer_label])

