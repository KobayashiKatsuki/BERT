import torch
from transformers import BertJapaneseTokenizer, BertModel

# GPUの確認
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name())

# 教科書第4章

# トークナイザ
model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
tokenizer  = BertJapaneseTokenizer.from_pretrained(model_name)

# トークン化　文章をトークン列に変換する
print(tokenizer.tokenize('これは俺の物語だ'))
# ['これ', 'は', '俺', 'の', '物語', 'だ']
print(tokenizer.tokenize('ファルシのルシがコクーンでパージ'))
# ['ファル', '##シ', 'の', 'ルシ', 'が', 'コク', '##ーン', 'で', 'パー', '##ジ']

# 符号化　文章をトークンのID列に変換
input_ids =  tokenizer.encode('誰かを助けるのに理由がいるかい？')
print(input_ids)
# [2, 3654, 29, 11, 13302, 5602, 1515, 14, 33, 3976, 2935, 3]
print(tokenizer.convert_ids_to_tokens(input_ids))
# ['[CLS]', '誰', 'か', 'を', '助ける', 'のに', '理由', 'が', 'いる', 'かい', '?', '[SEP]']

# padding処理
# tokenizerを関数として呼び出すことで実行可能
text = 'だったら壁にでも話していろ'
encoding = tokenizer(
    text, max_length=12, padding='max_length', truncation=True
)
print(encoding)
# {
# 'input_ids': [2, 308, 3318, 3057, 7, 962, 4341, 16, 8849, 3, 0, 0],
# 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
# }

print(tokenizer.convert_ids_to_tokens(encoding['input_ids']))
# ['[CLS]', 'だっ', 'たら', '壁', 'に', 'でも', '話し', 'て', 'いろ', '[SEP]', '[PAD]', '[PAD]']

# 複数まとめて処理
# padding='longest'にすると一番長い系列に合わせる
text_list = ['クックック、黒マテリア', 'クックック、メテオ呼ぶ']
encodint = tokenizer(
    text_list, padding='longest'
)
print(encodint)


# transformersのBERTへの入力はpytorchのtorch.Tensorクラスなので
# そうなるよう変換するには
# return_tensors='pt'オプションを入れる

encoding = tokenizer(
    text_list,
    max_length=10,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)
print(encoding)



