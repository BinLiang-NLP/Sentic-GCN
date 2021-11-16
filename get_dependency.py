# -*- coding: utf-8 -*-

import numpy as np
import spacy
import pickle

nlp = spacy.load('en_core_web_sm')


def dependency(text):
    # https://spacy.io/docs/usage/processing-text
    document = nlp(text)
    print(document)
    print('#'*30)
    
    for token in document:
        # print('father:',token)
        for child in token.children:
            print('father:', token)
            print('child:', child)
        print('='*30)


if __name__ == '__main__':
    text = 'Fun in the emergency room.'
    dependency(text)
