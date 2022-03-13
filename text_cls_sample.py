"""
参考
http://mathshingo.chillout.jp/blog30.html
https://qiita.com/m__k/items/e312ddcf9a3d0ea64d72
https://stackoverflow.com/questions/66302371/how-to-specify-the-loss-function-when-finetuning-a-model-using-the-huggingface-t
https://towardsdatascience.com/fine-tuning-pretrained-nlp-models-with-huggingfaces-trainer-6326a4456e7b
"""

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

from transformers import BertJapaneseTokenizer, BertModel
from transformers import TrainingArguments, Trainer

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix

MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'


"""
タスク用Dataset
"""
class PosiNegaDataset(Dataset):
  def __init__(self, encodings, labels=None):
    self.encodings = encodings
    self.labels = labels

  def __len__(self):
    return len(self.labels)
  
  def __getitem__(self, idx):
    item = { k: torch.tensor(v[idx]) for k, v in self.encodings.items() }
#         item = { k: torch.tensor(v[idx]).cuda() for k, v in self.encodings.items() }
    if self.labels is not None:
      item["labels"] = torch.tensor(self.labels[idx])
    return item


"""
BERTによる分類を行うレイヤのクラス
"""
class BertClassifier(nn.Module):
  def __init__(self, pretrained_model):
    super(BertClassifier, self).__init__()
    
    # 事前学習モデル
    self.model = pretrained_model
    
    # 線形変換層（全結合層）
    # ポジネガ（２カテゴリ）分類なので出力層は2
    self.classifier = nn.Linear(in_features=768, out_features=2)
    
    # 重み初期化
    nn.init.normal_(self.classifier.weight, std=0.02)
    nn.init.normal_(self.classifier.bias, 0)

  # def forward(self, input_ids):
  def forward(self, input_ids, labels, token_type_ids=None, attention_mask=None):
    output = self.model(input_ids)

    # トークン毎の特徴量（使わない）
    # last_hidden_state = output.last_hidden_state
    
    # 文代表（[CLS]トークン）の特徴量
    pooler_output = output.pooler_output
            
    # 分類タスク実行結果
    # -> Trainerのcompute_lossに渡される
    output_classifier = self.classifier(pooler_output)
    
    return output_classifier        


"""
タスク用Trainer
自作モデルのforward()の結果を受けてloss計算する
"""
class PosiNegaTrainer(Trainer):
  def compute_loss(self, model, inputs, return_outputs=False):
    # truth labels
    labels = inputs.get("labels")

    # forward pass
    outputs = model(**inputs)

    # compute custom loss
    loss_fct = nn.CrossEntropyLoss()
    # loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
    loss = loss_fct(outputs.view(-1, 2), labels.view(-1))

    return (loss, outputs) if return_outputs else loss



if __name__ == '__main__':
  # print(torch.cuda.is_available())

  # トークナイザ
  tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)
  # 事前学習モデル
  model = BertModel.from_pretrained(MODEL_NAME)
  # model = model.cuda()  

  # データセット用意
  df_dataset = pd.read_csv(
    'D:/DataSet/chABSA-dataset/chABSA-dataset/dataset.tsv',
    sep='\t', 
    header=None
  ).rename(columns={0:'text', 1:'label'}).loc[:, ['text', 'label']]

  # 特徴量X、ラベルyを取得
  X, y = df_dataset["text"].values, df_dataset["label"].values

  # train, val, test分割
  # random_stateはシャッフルの乱数シード固定、stratifyは正例、負例のラベル数均一にする処理
  X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
  X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=0, stratify=y_val)

  # トークナイザでモデルへのinputとなるようencodingする
  max_len = 256 #512

  enc_train = tokenizer(
    X_train.tolist(), 
    add_special_tokens=True, 
    max_length=max_len,
    padding='max_length',
    return_tensors='pt',
  )
  # tokenizer.convert_ids_to_tokens(enc_train['attention_masks'].tolist()[0])

  enc_val = tokenizer(
    X_val.tolist(), 
    add_special_tokens=True, 
    max_length=max_len,
    padding='max_length',
    return_tensors='pt',
  )

  enc_test = tokenizer(
    X_test.tolist(), 
    add_special_tokens=True, 
    max_length=max_len,
    padding='max_length',
    return_tensors='pt',
  )

  # Datasetを作成
  ds_train = PosiNegaDataset(enc_train, y_train)
  ds_val = PosiNegaDataset(enc_val, y_val)
  ds_test = PosiNegaDataset(enc_test, y_test)

  # DataLoaderを作成
  batch_size = 8
  dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
  dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

  # ファインチューニングするBERTモデル
  bert_classifier = BertClassifier(model)

  # Trainerを作成
  training_args = TrainingArguments(
    output_dir='./outputs',
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    no_cuda=False,
  )

  trainer = PosiNegaTrainer(
    model=bert_classifier,
    args=training_args,
    train_dataset=ds_train,
    eval_dataset=ds_val,
  )

  # ファインチューニング
  trainer.train()