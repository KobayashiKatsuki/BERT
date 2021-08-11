import numpy as np
import torch
from torch._C import TracingState
from transformers import BertJapaneseTokenizer, BertForMaskedLM

def main():
    # BERTモデル他
    model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
    tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
    bert_mlm = BertForMaskedLM.from_pretrained(model_name)
    bert_mlm = bert_mlm.cuda()
    
    # 穴埋めしたい文章
    # text = 'この世で最も恐ろしいもの、それは[MASK]である。'
    text = '人生とは、[MASK]である。'
    # text = '今日も一日、[MASK]。明日はもっと[MASK]。'
    
    print(tokenizer.tokenize(text))
    
    input_ids = tokenizer.encode(text, return_tensors='pt')
    input_ids = input_ids.cuda() # データをGPUに乗せるときはこれ！

    # BERTに入力し分類スコアを得る
    # 推論のみ実施するとき pytorch の no_grad() 
    # [MASK]が複数ある場合は貪欲法やビームサーチが用いられる。実装頑張れ。
    with torch.no_grad():

        output = bert_mlm(input_ids=input_ids) # 1文書のみなので系列長の指定不要
        scores= output.logits # これがMASKに対する各トークンのスコアを格納したテンソル！
        # スコアのテンソルは [ バッチサイズ, 系列長, 語彙数 ]
        print(scores.size())
    
        # MASKの位置を調べる
        # 文書は0番目（1文書しかないので）
        mask_position =input_ids[0].tolist().index(4) # id=4（[MASK]のID）の要素番号を取得

        # 上位kの結果を得る
        num_k = 5
        topk = scores[0, mask_position].topk(num_k)
        # topk(
        #     indices=tensor([ 9867, 26241,   446,     1,  7627], device='cuda:0'), # トークンID
        #     values=tensor([11.6924, 10.1412, 10.0819,  9.8133,  9.6616], device='cuda:0') # トークンのスコア
        # )

        # 結果取得
        ids_topk = topk.indices
        tokens_topk = tokenizer.convert_ids_to_tokens(ids_topk)
        scores_topk = topk.values.cpu().detach().numpy().copy()

        unmasked_text_topk = []
        for i, token in enumerate(tokens_topk):
            token = token.replace('##', '')
            unmasked_text_topk.append({'score': scores_topk[i], 'text': text.replace('[MASK]', token, 1)})
        

        # 結果表示
        print('result: ----------')
        for t in unmasked_text_topk:
            print(f'{t["score"]}\t{t["text"]}')




def predict_mask_topk(text, tokenizer, bert_mlm, num_topk):
    """
    与えられたtextの[MASK]をスコア上位のトークンで置き換える
    """
    # テキストをBERTへの入力形式ID列にする
    input_ids = tokenizer.encode(text, return_tensors='pt')
    input_ids = input_ids.cuda()
    
    # ID列をBERT MLMモデルに入れて系列内各要素のスコアを得る
    with torch.no_grad():
        output = bert_mlm(input_ids=input_ids)
    scores= output.logits

    # MASKの位置のスコア上位を得る
    mask_position =input_ids[0].tolist().index(4) # id=4（[MASK]のID）の要素番号を取得
    topk = scores[0, mask_position].topk(num_topk)
    # IDs, Scores
    ids_topk = topk.indices
    scores_topk = topk.values.cpu().detach().numpy().copy()
    # Tokens
    tokens_topk = tokenizer.convert_ids_to_tokens(ids_topk)

    # MASKを置き換えて出力文章を生成
    unmasekd_text_topk = []
    for token in tokens_topk:
        token = token.replace('##', '')
        unmasekd_text_topk.append(text.replace('[MASK]', token, 1))

    return unmasekd_text_topk, scores_topk



def bert_mlm_with_beam_search(text, beam_lenght=10) -> list:
    """
    ビームサーチ法による文書穴埋め
    """

    # BERTモデル他
    model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
    tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
    bert_mlm = BertForMaskedLM.from_pretrained(model_name)
    bert_mlm = bert_mlm.cuda()

    print(tokenizer.tokenize(text))
    
    # beam searchの結果格納用 
    beam_search_result = [(text, 0)]

    # MASKが無くなるまで処理
    mask_count = text.count('[MASK]')

    for _ in range(mask_count):

        # 現時点で得られている上位のテキストそれぞれで処理
        tmp_result = []
        for t, s in beam_search_result:
            unmasked_text_list, scores_topk = predict_mask_topk(t, tokenizer, bert_mlm, beam_lenght)
            # 各テキストでMASK外した結果topkをビームサーチの結果に追加していく
            one_text_topk = []
            for tk, sk in zip(unmasked_text_list, scores_topk):
                one_text_topk.append((tk, s+sk))
            tmp_result.extend(one_text_topk)
        beam_search_result.extend(tmp_result)

        # 現時点で得られているテキストのうちスコア上位のみ残す
        beam_search_result = sorted(beam_search_result, reverse=True, key=lambda x: x[1])
        beam_search_result = beam_search_result[:beam_lenght]

    return beam_search_result


if __name__ == '__main__':
    # main()

    # 演習問題
    # 次の文章の穴埋めをBERT+ビームサーチ法で行う関数を実装せよ
    # text = '俺が[MASK]の時、あいつは[MASK]だった。'
    text = '今日は[MASK][MASK]へ行く。'

    unmasked_text_list = bert_mlm_with_beam_search(text, beam_lenght=10)

    print('result: ----------')
    for t in unmasked_text_list:
        print(t[0])
