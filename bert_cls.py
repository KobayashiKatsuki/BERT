import os
import random
import glob
from re import A
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer, BertForSequenceClassification
import pytorch_lightning as pl
MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'

from transformers import AutoTokenizer

from bert_for_sequence_classification_pl import BertForSequenceClassification_pl


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # 分類モデル　num_labelsは分類カテゴリ数（ネガポジの2値分類なので2）
    # BertForSequenceClassificationは
    # BertModelの[CLS]トークンに対応する出力に変換層をかませたものを出力している
    bert_sc = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    bert_sc = bert_sc.cuda()

    # テストデータ
    text_list = [
        'この映画は最高の出来だった。',
        'この映画は今までで一番ひどかった。',
        'この映画を見ていない人は人生損している。',
    ]
    label_list = [1, 0, 1]

    encoding = tokenizer(
        text_list,
        padding = 'longest',
        return_tensors='pt'
    ) 
    encoding['labels'] = torch.tensor(label_list) # 入力にラベルを加えるとLoss計算可能に
    encoding = {k: v.cuda() for k, v in encoding.items()}
    labels = torch.tensor(label_list).cuda() # labelもGPUに載せる

    with torch.no_grad():
        output = bert_sc.forward(**encoding)
        # output = bert_sc(**encoding)
    scores = output.logits # 分類スコア
    print(scores)
    # まだ単なる出力値でしかない　
    # 学習するならこのあとSoftmax掛けるとかする

    labels_predicted = scores.argmax(-1) # 最大スコアのラベルを得る
    print(labels_predicted)

    # 損失
    loss = output.loss
    print(loss)


def dataloader_test():
    """
    DataLoaderについて
    DataLoaderはミニバッチ作成に有用
    """
    # フォーマット　
    # 各データを表す辞書を要素とした配列
    dataset_for_loader = [
        {'data': torch.tensor([0, 1]), 'labels': torch.tensor(0)},
        {'data': torch.tensor([2, 3]), 'labels': torch.tensor(1)},
        {'data': torch.tensor([4, 5]), 'labels': torch.tensor(2)},
        {'data': torch.tensor([6, 7]), 'labels': torch.tensor(3)},
    ]
    # それをDataLoaderのコンストラクタに渡す　バッチサイズも指定 shuffleをTrueにしないと先頭から取り出す
    loader = DataLoader(dataset_for_loader, batch_size=2, shuffle=True)

    # ミニバッチ取り出し
    # データ構造は同じで'data'に各データ格納したテンソル、'labels'に各データ格納テンソル
    for idx, batch in enumerate(loader):
        print(f'# batch {idx}')
        print(batch)
        #
        # ファインチューニングではここでミニバッチ毎の処理を行う
        #

def data_preprocess():
    """
    ファインチューニングのデータ前処理
    """

    # カテゴリリスト
    category_list = [
        'dokujo-tsushin',
        'it-life-hack',
        'kaden-channel',
        'livedoor-homme',
        'movie-enter',
        'peachy',
        'smax',
        'sports-watch',
        'topic-news'
    ]

    # 以下テンプレ
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    max_lenght = 128
    dataset_for_loader = []

    # 文書のトークナイズとラベルの付与（カテゴリ名の要素番号をラベルとしている）
    for label, category in enumerate(tqdm(category_list)): # tqdmはプログレスバー表示処理モジュール
        # globは正規表現でファイルアクセス可能なモジュール
        for file in glob.glob(f'../../DataSet/livedoor-news-dataset/{category}/{category}*'):
        # for file in glob.glob(f'../../DataSet/livedoor-news-dataset-mini/{category}/{category}*'):
            lines = open(file, encoding='utf-8').read().splitlines() # UTF-8なのでエンコード指定
            text = '\n'.join(lines[3:]) #４行目以降を改行で結合して抜き出す

            # い　つ　も　の
            # と思いきや return_tensor='pt'は指定しない 単一文書を1階のテンソルで扱いたいから？バッチ処理の関係？
            # ともかく、DataLoaderを介するときは2次元テンソルにせず１次元テンソルのままにしておく
            encoding = tokenizer(
                text,
                max_length=max_lenght,
                padding='max_length',
                truncation=True,
            )
            encoding['labels'] = label
            encoding = { k: torch.tensor(v) for k, v in encoding.items() }
            dataset_for_loader.append(encoding)

    #print(dataset_for_loader[0])

    # 学習・検証・テストにデータ分割
    random.shuffle(dataset_for_loader)
    n = len(dataset_for_loader)
    n_train = int(0.6*n) # 60%を学習に
    n_valid = int(0.2*n) # 20％を検証、残りをテストに

    dataset_train = dataset_for_loader[:n_train]
    dataset_valid = dataset_for_loader[n_train:n_train+n_valid]
    dataset_test = dataset_for_loader[n_train+n_valid:]
    
    # DataLoaderをようやく作成
    dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=256)
    dataloader_test = DataLoader(dataset_test, batch_size=256)

    return dataloader_train, dataloader_valid, dataloader_test



def bert_finetuning( dataloader_train: DataLoader, dataloader_valid: DataLoader, dataloader_test: DataLoader):
    """
    ファインチューニング
    """

    # BERTモデル
    model = BertForSequenceClassification_pl(MODEL_NAME, num_labels=9, lr=1e-5)

    # 学習時のモデルの重み保存条件
    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_weights_only=True,
        dirpath='model/'
    )

    # PyTorch Lightningログ設定
    logger = TensorBoardLogger(
        save_dir=os.getcwd(),
        name='logs'
    )
    
    # 学習の方法を指定
    # ★ 学習にはTrainerクラスを使用

    # ★★　ただしこれは PyTorch Lightning 版の Trainer ★★
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=10,
        callbacks=[checkpoint],
        logger=logger
    )

    # ファインチューニング実行
    trainer.fit(model, dataloader_train, dataloader_valid)

    # 精度評価
    test = trainer.test(test_dataloaders=dataloader_test)
    print(f'Accuracy: {test[0]["accuracy"]:.2f}')
    
    # モデルの保存
    model = BertForSequenceClassification_pl.load_from_checkpoint(
        checkpoint.best_model_path
    )
    model.bert_sc.save_pretrained('./model_transformers')



def predict_document_category(dataloader_test: DataLoader):
    """ファインチューニング済モデルを用いた文書分類タスク
    """

    # モデルのロード
    bert_sc = BertForSequenceClassification.from_pretrained('./model_transformers')
    bert_sc = bert_sc.cuda()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    max_lenght = 128


    # カテゴリリスト
    category_list = [
        'dokujo-tsushin',
        'it-life-hack',
        'kaden-channel',
        'livedoor-homme',
        'movie-enter',
        'peachy',
        'smax',
        'sports-watch',
        'topic-news'
    ]

    # タスクの対象文書
    # うまいことBERTの入力フォーマットに合わせるよう頑張れ
    text_list = []
    label_list = []
    for label, category in enumerate(tqdm(category_list)):
        for file in glob.glob(f'../../DataSet/livedoor-news-dataset-mini/{category}/{category}*'):
            lines = open(file, encoding='utf-8').read().splitlines()
            text = '\n'.join(lines[3:])

            text_list.append(text)
            label_list.append(label)

    encoding = tokenizer(
        text_list,
        max_length=max_lenght,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ) 
    encoding['labels'] = torch.tensor(label_list)
    encoding = { k: v.cuda() for k, v in encoding.items() }
    labels = torch.tensor(label_list).cuda()

    # ファインチューニング後のBERTモデルで文書分類実行
    with torch.no_grad():
        output = bert_sc(**encoding)
    # 分類結果ラベル
    labels_predicted = output.logits.argmax(-1)


    # 結果表示
    for idx, (label, text) in enumerate(zip(label_list, text_list)):
        headline = text.split('\n')[0]

        print(f'text: \n{headline}')
        print(f'Label: {label}\tPredicted: {labels_predicted[idx].item()}')
        print('----------------------------')


    # 精度
    num_accurate = (labels_predicted == labels).sum().item()
    print(num_accurate / labels.size(0))
    




if __name__ == '__main__':
    #main()
    #dataloader_test()

    #dataloader_train, dataloader_valid, dataloader_test = data_preprocess()
    #bert_finetuning(dataloader_train, dataloader_valid, dataloader_test)

    predict_document_category(dataloader_test)

