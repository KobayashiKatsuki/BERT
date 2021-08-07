import torch
from torch._C import TracingState
from transformers import BertJapaneseTokenizer, BertModel

def main():
    # BERTモデル
    model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
    tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
    bert = BertModel.from_pretrained(model_name)

    # GPUに乗っける
    bert = bert.cuda()
    #print(bert.config)
    
    # BertConfig {
    #   "_name_or_path": "cl-tohoku/bert-base-japanese-whole-word-masking",
    #   "architectures": [
    #     "BertForMaskedLM"
    #   ],
    #   "attention_probs_dropout_prob": 0.1,
    #   "gradient_checkpointing": false,
    #   "hidden_act": "gelu",
    #   "hidden_dropout_prob": 0.1,
    #   "hidden_size": 768,
    #   "initializer_range": 0.02,
    #   "intermediate_size": 3072,
    #   "layer_norm_eps": 1e-12,
    #   "max_position_embeddings": 512,
    #   "model_type": "bert",
    #   "num_attention_heads": 12,
    #   "num_hidden_layers": 12,
    #   "pad_token_id": 0,
    #   "position_embedding_type": "absolute",
    #   "tokenizer_class": "BertJapaneseTokenizer",
    #   "transformers_version": "4.8.2",
    #   "type_vocab_size": 2,
    #   "use_cache": true,
    #   "vocab_size": 32000
    # }

    # 上記の主な部分の見方は
    # num_hidden_layers(中間層): 12層
    # hidden_size(BERTの出力次元数): 768
    # max_position_embedding(入力可能な最大トークン長): 512


    # 一度に処理可能な文書量 = バッチサイズ
    # 以下は2文書（バッチサイズ2）の処理過程

    text_list = [
        'バッチサイズとは一度に処理する文書の量',
        '以下はバッチサイズ2の処理過程である'
    ]

    # まず符号化
    encoding = tokenizer(
        text_list,
        max_length=32,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # データをGPUに乗っける
    encoding = { k: v.cuda() for k, v in encoding.items() }

    # BERTへ入出力
    output = bert(**encoding)
    
    # 最終層を出力、テンソルとして取得する
    last_hidden_state = output.last_hidden_state
    print(last_hidden_state) # テンソルそのもの
    print(last_hidden_state.size()) # テンソルサイズ
    # torch.Size([2, 32, 768])  [ バッチサイズ, 系列長, 特徴ベクトル次元数 ]

    # この状態では last_hidden_state もGPUに乗っているので
    # CPUに移すには
    last_hidden_state = last_hidden_state.cpu()
    # numpay.ndarrayに変換 detachで切り離し、copyで参照分ける
    last_hidden_state = last_hidden_state.detach().numpy().copy()
    print(last_hidden_state)
    # listに変換
    last_hidden_state = last_hidden_state.tolist()
    #print(last_hidden_state)




if __name__ == '__main__':
    main()
