import itertools
import random
import json
from turtle import width
from tqdm import tqdm
import numpy as np
import unicodedata

import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer, BertForTokenClassification

from pprint import pprint

MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'

# Wikipedia学習データ
# https://raw.githubusercontent.com/stockmarkteam/ner-wikipedia-dataset/main/ner.json

def test1():
    text = '昨日のみらい事務所との打ち合わせは順調だった。'
    entities = [
        {'name': 'みらい事務所', 'span': [3,9], 'type_id': 1},
    ]
    entities = sorted(entities, key=lambda x: x['span'][0])
    splitted = [] # 分割後の文字列を追加していく
    position = 0
    for entity in entities:
        start = entity['span'][0]
        end = entity['span'][1]
        label = entity['type_id']
        # 固有表現ではないものには0のラベルを付与
        splitted.append({'text':text[position:start], 'label':0}) 
        # 固有表現には、固有表現のタイプに対応するIDをラベルとして付与
        splitted.append({'text':text[start:end], 'label':label}) 
        position = end
    splitted.append({'text': text[position:], 'label':0})
    splitted = [ s for s in splitted if s['text'] ] # 長さ0の文字列は除く

    print(splitted, width=40)


def test2():
    tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)
    text = "騰訊の英語名はTencent Holdings Ltdである。"

    words = tokenizer.word_tokenizer.tokenize(text)
    tokens = []

    for word in words:
        tokens_word = tokenizer.subword_tokenizer.tokenize(word) 
        print(tokens_word)
        tokens.extend(tokens_word)

    # pprint(tokens, width=40)



if __name__ == '__main__':
    test2()
