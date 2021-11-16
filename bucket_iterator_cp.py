# -*- coding: utf-8 -*-

import math
import random
import torch
import numpy

class BucketIterator(object):
    def __init__(self, data, batch_size, sort_key='text_indices', shuffle=True, sort=True):
        self.shuffle = shuffle
        self.sort = sort
        self.sort_key = sort_key
        self.batches = self.sort_and_pad(data, batch_size)
        self.batch_len = len(self.batches)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        if self.sort:
            sorted_data = sorted(data, key=lambda x: len(x[self.sort_key]))
        else:
            sorted_data = data
        batches = []
        for i in range(num_batch):
            batches.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size]))
        return batches

    def pad_data(self, batch_data):
        batch_context = []
        batch_aspect = []
        batch_text_indices = []
        batch_context_indices = []
        batch_aspect_indices = []
        batch_left_indices = []
        batch_polarity = []
        batch_dependency_graph = []
        batch_sentic_graph = []
        batch_sdat_graph = []
        max_len = max([len(t[self.sort_key]) for t in batch_data])
        for item in batch_data:
            context, aspect, text_indices, context_indices, aspect_indices, left_indices,\
            polarity, dependency_graph, sentic_graph, sdat_graph = \
                item['context'], item['aspect'], \
                item['text_indices'], item['context_indices'], item['aspect_indices'], item['left_indices'],\
                item['polarity'], item['dependency_graph'], item['sentic_graph'], item['sdat_graph']
            text_padding = [0] * (max_len - len(text_indices))
            context_padding = [0] * (max_len - len(context_indices))
            aspect_padding = [0] * (max_len - len(aspect_indices))
            left_padding = [0] * (max_len - len(left_indices))
            batch_context.append(context)
            batch_aspect.append(aspect)
            batch_text_indices.append(text_indices + text_padding)
            batch_context_indices.append(context_indices + context_padding)
            batch_aspect_indices.append(aspect_indices + aspect_padding)
            batch_left_indices.append(left_indices + left_padding)
            batch_polarity.append(polarity)
            batch_dependency_graph.append(numpy.pad(dependency_graph, 
                ((0,max_len-len(text_indices)),(0,max_len-len(text_indices))), 'constant'))
            batch_sentic_graph.append(numpy.pad(sentic_graph, 
                ((0,max_len-len(text_indices)),(0,max_len-len(text_indices))), 'constant'))
            batch_sdat_graph.append(numpy.pad(sdat_graph, 
                ((0,max_len-len(text_indices)),(0,max_len-len(text_indices))), 'constant'))

        #for i in range(len(batch_dependency_graph)):
            #print('dependency_graph:', batch_dependency_graph[i].shape)
            #print('sentic_graph:', batch_sentic_graph[i].shape)

        return { \
                'context': batch_context, \
                'aspect': batch_aspect, \
                'text_indices': torch.tensor(batch_text_indices), \
                'context_indices': torch.tensor(batch_context_indices), \
                'aspect_indices': torch.tensor(batch_aspect_indices), \
                'left_indices': torch.tensor(batch_left_indices), \
                'polarity': torch.tensor(batch_polarity), \
                'dependency_graph': torch.tensor(batch_dependency_graph), \
                'sentic_graph': torch.tensor(batch_sentic_graph), \
                'sdat_graph': torch.tensor(batch_sdat_graph)
            }

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for idx in range(self.batch_len):
            yield self.batches[idx]
