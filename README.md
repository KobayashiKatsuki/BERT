## memo

#### 学習（ファインチューニング）
* fit: pytorch_lightningのTrainerクラスのメソッド
* train: Hugging Face transformersのTrainerクラスのメソッド
* フルスクラッチ：forwardとbackwordで頑張る

plは書籍で使われているけどあんまり一般的でない？
hugging face Trainer覚えよう

#### BERTを扱うクラス設計

pre-trainedモデル～Fine Tuningまで含めるのが一般的

``` python

from torch import nn

class BertClassifier(nn.Model): # トレーニングしやすいようにnn.Model継承

  def __init__(self, pretrained_model):
    super(BertClassifier, self).__init__() 

    # 事前学習モデル
    self.model = pretrained_model

    # タスク毎に定義する線形変換層
    # 分類タスクならカテゴリ数(↓は9カテゴリ)のノードの層
    self.classifier = nn.Linear(in_features=768, out_features=9)
    
    # 重み初期化
    # パス

  def forward(self, input_ids):
    # 入力トークンをBERTにつっこんで出力を得る
    output = self.model(input_ids)

    # タスクによって何するかきめる
    # last_hidden_state使う場合も
    feature_pooler = output.pooler_output
    # ちなみに
    # output=(last_layerの全てのtoken,last_layerの最初のToken(=[CLS]Token)) 
    # らしい

    # タスクの出力を得る
    task_output = self.classifier(feature_pooler)
    return task_output

```