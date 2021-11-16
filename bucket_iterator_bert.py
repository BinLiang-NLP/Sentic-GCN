# -*- coding: utf-8 -*-

import math
import random
import torch
import numpy

class BucketIterator(object):
    def __init__(self, data, batch_size, sort_key='sdat_graph', shuffle=True, sort=False):
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
            #for info in sorted_data[i*batch_size:(i+1)*batch_size]:
                #print(info)
                #print('#'*30)
            batches.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size]))
        return batches

    def pad_data(self, batch_data):
        batch_bert_text_indices = []
        batch_bert_seg_indices = []
        batch_bert_raw_indices = []
        batch_bert_aspect_indices = []

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
        max_text_len = max([len(t['text_indices']) for t in batch_data])
        max_text_len = 80
        max_pad = 0
        for item in batch_data:
            text_bert_indices, bert_segments_ids, text_raw_bert_indices, aspect_bert_indices,\
            context, aspect, text_indices, context_indices, aspect_indices, left_indices,\
            polarity, dependency_graph, sentic_graph, sdat_graph = \
                item['text_bert_indices'], item['bert_segments_ids'], item['text_raw_bert_indices'], item['aspect_bert_indices'],\
                item['context'], item['aspect'], \
                item['text_indices'], item['context_indices'], item['aspect_indices'], item['left_indices'],\
                item['polarity'], item['dependency_graph'], item['sentic_graph'], item['sdat_graph']

            bert_text_padding = [0] * (max_len - len(text_bert_indices) + max_pad)
            bert_seg_padding = [0] * (max_len - len(bert_segments_ids) + max_pad)
            bert_raw_padding = [0] * (max_len - len(text_raw_bert_indices) + max_pad)
            bert_aspect_padding = [0] * (max_len - len(aspect_bert_indices) + max_pad)

            text_padding = [0] * (max_len - len(text_indices))
            context_padding = [0] * (max_len - len(context_indices))
            aspect_padding = [0] * (max_len - len(aspect_indices))
            left_padding = [0] * (max_len - len(left_indices))

            #print('max_len:', max_len)

            #print('bert_text_padding:', bert_text_padding)

            batch_bert_text_indices.append(text_bert_indices)
            batch_bert_seg_indices.append(bert_segments_ids)
            batch_bert_raw_indices.append(text_raw_bert_indices)
            batch_bert_aspect_indices.append(aspect_bert_indices)

            batch_context.append(context)
            batch_aspect.append(aspect)
            batch_text_indices.append(text_indices)
            batch_context_indices.append(context_indices)
            batch_aspect_indices.append(aspect_indices)
            batch_left_indices.append(left_indices)
            batch_polarity.append(polarity)
            #batch_dependency_graph.append(numpy.pad(dependency_graph, 
                #((0,max_len-len(text_indices)+max_pad),(0,max_len-len(text_indices)+max_pad)), 'constant'))
            #batch_sentic_graph.append(numpy.pad(sentic_graph, 
                #((0,max_len-len(text_indices)+max_pad),(0,max_len-len(text_indices)+max_pad)), 'constant'))
            #batch_sdat_graph.append(numpy.pad(sdat_graph, 
                #((0,max_text_len-len(text_indices)),(0,max_text_len-len(text_indices))), 'constant'))
            batch_sdat_graph.append(sdat_graph)

        #for i in range(len(batch_bert_text_indices)):
            #print('batch_bert_aspect_indices:', batch_bert_text_indices[i])
            #print('-'*20)
            #print('sentic_graph:', batch_sdat_graph[i])
            #print('='*20)

        return { \
                'text_bert_indices': torch.tensor(batch_bert_text_indices), \
                'bert_segments_ids': torch.tensor(batch_bert_seg_indices), \
                'text_raw_bert_indices': torch.tensor(batch_bert_raw_indices), \
                'aspect_bert_indices': torch.tensor(batch_bert_aspect_indices), \

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
