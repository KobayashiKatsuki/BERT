import torch
from torch.functional import Tensor
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer, BertForSequenceClassification
import pytorch_lightning as pl

class BertForSequenceClassification_pl(pl.LightningModule):
    """
    モデルの振る舞いを記述するクラス
    PyTorch Lightningを継承する
    モデルやデータをGPUに載せる操作も自動でやってくれる
    """
    
    def __init__(self, model_name, num_labels, lr):
        super().__init__()
        # ハイパラ保存
        self.save_hyperparameters()
        # BERTモデル
        self.bert_sc = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # 学習データのミニバッチ'batch'を与えると損失を返す関数
    # DataLoaderをfor enumで回せば勝手にバッチになる
    # PyTorch Lightningではパラメータ更新も勝手にやってくれる
    def training_step(self, batch, batch_idx):
        output = self.bert_sc(**batch) # BERTへの入出力はPyTorchテンソル放り込めばOK
        loss = output.loss
        self.log('train_loss', loss)
        return loss

    # 検証データのミニバッチを与えると検証データ評価計算する関数
    def validation_step(self, batch, batch_idx):
        output = self.bert_sc(**batch)
        val_loss = output.loss
        self.log('val_loss', val_loss)

    # テストデータのミニバッチを与えると精度評価する関数    
    def test_step(self, batch, batch_idx):
        labels = batch.pop('labels') # 正解ラベル
        output = self.bert_sc(**batch)
        labels_predicted = output.logits.argmax(-1) # 推論結果ラベル（スコア(logits)が最も高い要素番号）

        # 演算 (tensorオブジェクト == tensorオブジェクト)　は
        # 一致する要素をTrue、そうでない要素をFalseとして格納した<<Tensor>>を返す
        # Tensor.item() はテンソルの要素を取得する
        num_correct = (labels_predicted == labels).sum().item()

        accuracy = num_correct / labels.size(0)
        self.log('accuracy', accuracy)
        
    # 学習に用いるオプティマイザを返す関数
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)



if __name__ == '__main__':
    t1 = torch.tensor([[1, 0], [1, 2]])
    print(t1.argmax(-1))

