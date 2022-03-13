import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

from transformers import BertJapaneseTokenizer, BertForSequenceClassification
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
    # item = { k: torch.tensor(v[idx]) for k, v in self.encodings.items() }
    item = { k: torch.tensor(v[idx], device=torch.device('cuda')) for k, v in self.encodings.items() }
    if self.labels is not None:
      # item["labels"] = torch.tensor(self.labels[idx])
      item["labels"] = torch.tensor(self.labels[idx], device=torch.device('cuda'))

    return item


  
# compute_matricsを定義
def compute_metrics(p):
  pred, labels = p
  pred = np.argmax(pred, axis=1)
  
  accuracy = accuracy_score(y_true=labels, y_pred=pred)
  recall = recall_score(y_true=labels, y_pred=pred)
  precision = precision_score(y_true=labels, y_pred=pred)
  f1 = f1_score(y_true=labels, y_pred=pred)
  
  return {"accuracy": accuracy, "recall": recall, "precision": precision, "f1": f1}



def BertFineTuning():
  """
  ファインチューニング
  """
  # トークナイザ
  tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)

  # 事前学習モデル
  model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
  model = model.cuda()  

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

  # Trainerを作成
  training_args = TrainingArguments(
    output_dir='./outputs',
    num_train_epochs=1,
    evaluation_strategy="steps",
    eval_steps=500,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_dir='./logs',
    logging_steps=10,
    no_cuda=False,
  )

  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_train,
    eval_dataset=ds_val,
    compute_metrics=compute_metrics,
  )

  # ファインチューニング
  trainer.train()

  # # 推論
  # with torch.no_grad():
  #   raw_pred, _, _ = trainer.predict(ds_test)

  # y_pred = np.argmax(raw_pred, axis=1)

  # accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
  # recall = recall_score(y_true=y_test, y_pred=y_pred)
  # precision = precision_score(y_true=y_test, y_pred=y_pred)
  # f1 = f1_score(y_true=y_test, y_pred=y_pred)

  # print(accuracy, recall, precision, f1)

  # cm = confusion_matrix(y_test, y_pred)
  # print(cm)



if __name__ == '__main__':
  # print(torch.cuda.is_available())
  BertFineTuning()
