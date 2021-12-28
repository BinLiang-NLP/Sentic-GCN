# -*- coding: utf-8 -*-

import os
import pickle
import torch
import torch.nn.functional as F
import numpy as np

from pytorch_pretrained_bert import BertModel
from data_utils_bert import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, ABSADataset, pad_and_truncate
from models import SenticGCN_BERT
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

        self.tokenizer = Tokenizer4Bert(self.opt.max_seq_len, self.opt.pretrained_bert_name)
        bert = BertModel.from_pretrained(opt.pretrained_bert_name)
        self.model = self.opt.model_class(bert, opt).to(opt.device)
        print('loading model {0} ...'.format(opt.model_name))
        self.model.load_state_dict(torch.load(opt.state_dict_path))
        self.model = self.model
        # switch model to evaluation mode
        self.model.eval()
        torch.autograd.set_grad_enabled(False)

    def evaluate(self, raw_text, aspect):
        senticNet = load_sentic_word()
        con_text = '[CLS] ' + raw_text.lower() + ' [SEP] ' + aspect.lower() + " [SEP]"
        #text_seqs = [self.tokenizer.text_to_sequence(raw_text.lower())]
        #aspect_seqs = [self.tokenizer.text_to_sequence(aspect.lower())]
        #left_seqs = [self.tokenizer.text_to_sequence(raw_text.lower().split(aspect.lower())[0])]

        text_indices = [self.tokenizer.text_to_sequence(raw_text.lower())]
        aspect_indices = [self.tokenizer.text_to_sequence(aspect.lower())]
        left_indices = [self.tokenizer.text_to_sequence(raw_text.lower().split(aspect.lower())[0])]

        left_len = np.sum(left_indices != 0)
        aspect_len = np.sum(aspect_indices != 0)
        text_len = np.sum(text_indices != 0)

        text_indices = torch.tensor(text_indices, dtype=torch.int64)
        aspect_indices = torch.tensor(aspect_indices, dtype=torch.int64)
        left_indices = torch.tensor(left_indices, dtype=torch.int64)

        concat_bert_indices = [self.tokenizer.text_to_sequence(con_text)]
        concat_segments_indices = [0] * (text_len + 2) + [1] * (aspect_len + 1)
        concat_segments_indices = [pad_and_truncate(concat_segments_indices, self.opt.max_seq_len)]
        concat_bert_indices = torch.tensor(concat_bert_indices, dtype=torch.int64)
        concat_segments_indices = torch.tensor(concat_segments_indices, dtype=torch.int64)

        sdat_graph = dependency_adj_matrix(raw_text.lower(), aspect.lower(), senticNet)
        sdat_graph = np.pad(sdat_graph, \
                    ((0,self.tokenizer.max_seq_len-sdat_graph.shape[0]),(0,self.tokenizer.max_seq_len-sdat_graph.shape[0])), 'constant')
        sdat_graph = torch.tensor([sdat_graph])

        data = {
            'text_bert_indices': concat_bert_indices,
            'text_indices': text_indices,
            'aspect_indices': aspect_indices,
            'bert_segments_indices': concat_segments_indices,
            'left_indices': left_indices,
            'sdat_graph': sdat_graph,
        }
        t_inputs = [data[col].to(opt.device) for col in self.opt.inputs_cols]
        t_outputs = self.model(t_inputs)

        t_probs = F.softmax(t_outputs, dim=-1).cpu().numpy()
        return t_probs



if __name__ == '__main__':
    dataset = 'rest14'
    # set your trained models here
    model_state_dict_paths = {
        'senticgcn_bert': 'state_dict/senticgcn_bert_'+dataset+'.pkl',
    }
    model_classes = {
        'senticgcn_bert': SenticGCN_BERT,
    }
    input_colses = {
        'senticgcn_bert': ['text_bert_indices', 'text_indices', 'aspect_indices', 'bert_segments_indices', 'left_indices', 'sdat_graph'],
    }
    class Option(object): pass
    opt = Option()
    opt.model_name = 'senticgcn_bert'
    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.dataset = dataset
    opt.state_dict_path = model_state_dict_paths[opt.model_name]
    opt.bert_dim = 768
    opt.hidden_dim = 300
    opt.polarities_dim = 3
    opt.max_seq_len = 85
    opt.pretrained_bert_name = 'bert-base-uncased'
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

