import os
import re

import cn2an
from pypinyin import lazy_pinyin, Style

from text.symbols import punctuation
from text.tone_sandhi import ToneSandhi

current_file_path = os.path.dirname(__file__)
pinyin_to_symbol_map = {
    line.split("\t")[0]: line.strip().split("\t")[1]
    for line in open(os.path.join(current_file_path, "opencpop-strict.txt")).readlines()
}#{"zuo"："z ou"}

import jieba.posseg as psg


rep_map = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "·": ",",
    "、": ",",
    "...": "…",
    "$": ".",
    "“": "'",
    "”": "'",
    "‘": "'",
    "’": "'",
    "（": "'",
    "）": "'",
    "(": "'",
    ")": "'",
    "《": "'",
    "》": "'",
    "【": "'",
    "】": "'",
    "[": "'",
    "]": "'",
    "—": "-",
    "～": "-",
    "~": "-",
    "「": "'",
    "」": "'",
}

tone_modifier = ToneSandhi()


def replace_punctuation(text):
    text = text.replace("嗯", "恩").replace("呣", "母")
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))

    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)

    replaced_text = re.sub(
        r"[^\u4e00-\u9fa5" + "".join(punctuation) + r"]+", "", replaced_text
    )

    return replaced_text


def g2p(text):
    pattern = r"(?<=[{0}])\s*".format("".join(punctuation))
    sentences = [i for i in re.split(pattern, text) if i.strip() != ""] #把“今天是个好天气!你说是不是?啊?”的整句进行分割["今天是个好天气!","你说是不是?","啊?"]这样的断句结果
    #[音素], [声调], [每个字的音素数量] 这里是当作sentences只有一句话写的示例，如果有多句话就会前后拼起来
    #["zh","e","zh","en","sh","i","yi","g","e","h","ao","r","en"], [4, 4, 2, 2, 4, 4, 1, 3, 3, 3, 3, 3, 3], [2, 2, 2, 1, 2, 2, 2]
    phones, tones, word2ph = _g2p(sentences)
    assert sum(word2ph) == len(phones)
    assert len(word2ph) == len(text)  # Sometimes it will crash,you can add a try-catch.
    #加上句子之间的间隔
    phones = ["_"] + phones + ["_"]
    tones = [0] + tones + [0]
    word2ph = [1] + word2ph + [1]
    return phones, tones, word2ph


def _get_initials_finals(word):
    initials = []
    finals = []
    orig_initials = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.INITIALS) #声母风格结果 ``z g``
    orig_finals = lazy_pinyin(
        word, neutral_tone_with_five=True, style=Style.FINALS_TONE3
    ) #韵母风格3 ``ong1 uo2``
    for c, v in zip(orig_initials, orig_finals):
        initials.append(c)
        finals.append(v)
    return initials, finals


def _g2p(segments): #传入多个句子["句子1","句子2"...]
    phones_list = []
    tones_list = []
    word2ph = []
    for seg in segments:
        # Replace all English words in the sentence
        seg = re.sub("[a-zA-Z]+", "", seg)
        seg_cut = psg.lcut(seg) #把一个完整的句子切割成[('我', n), ('喜欢', v), ('你', n)]
        initials = []
        finals = []
        seg_cut = tone_modifier.pre_merge_for_modify(seg_cut) #针对jieba的分词结果进行优化
        for word, pos in seg_cut:
            if pos == "eng":
                continue
            sub_initials, sub_finals = _get_initials_finals(word) #单个词汇转拼音返回(['zh', 'g'], ['ong1', 'uo2'])
            sub_finals = tone_modifier.modified_tone(word, pos, sub_finals) #继续对拼音结果进行连声的优化处理
            initials.append(sub_initials)
            finals.append(sub_finals)

            # assert len(sub_initials) == len(sub_finals) == len(word)
        initials = sum(initials, []) #[['zh', 'l'], ['sh'], ['zh', 'g']] => ['zh', 'l', 'sh', 'zh', 'g']
        finals = sum(finals, []) #[['e4', 'i3'], ['i4'], ['ong1', 'uo2']] => ['e4', 'i3', 'i4', 'ong1', 'uo2']

        for c, v in zip(initials, finals):
            raw_pinyin = c + v #合并出一个字的完整拼音+音调
            # NOTE: post process for pypinyin outputs
            # we discriminate i, ii and iii
            if c == v:
                assert c in punctuation
                phone = [c]
                tone = "0"
                word2ph.append(1)
            else:
                v_without_tone = v[:-1] #韵母部分
                tone = v[-1] #声调

                pinyin = c + v_without_tone #单个拼音
                assert tone in "12345"

                if c:
                    # 多音节，有声母+韵母组成，转换：h + uei -> hui
                    v_rep_map = {
                        "uei": "ui",
                        "iou": "iu",
                        "uen": "un",
                    }
                    if v_without_tone in v_rep_map.keys():
                        pinyin = c + v_rep_map[v_without_tone]
                else:
                    # 单音节，只有韵母组成，转换成完整读音
                    pinyin_rep_map = {
                        "ing": "ying",
                        "i": "yi",
                        "in": "yin",
                        "u": "wu",
                    }
                    if pinyin in pinyin_rep_map.keys():
                        pinyin = pinyin_rep_map[pinyin]
                    else:
                        single_rep_map = {
                            "v": "yu",
                            "e": "e",
                            "i": "y",
                            "u": "w",
                        }
                        if pinyin[0] in single_rep_map.keys():
                            pinyin = single_rep_map[pinyin[0]] + pinyin[1:]
                # 总之到这里pinyin变成了一个正确的没有声调的配音
                assert pinyin in pinyin_to_symbol_map.keys(), (pinyin, seg, raw_pinyin)
                phone = pinyin_to_symbol_map[pinyin].split(" ") #"z uo"->["z","uo"]，pinyin_to_symbol_map里面估计是全部的发音组合
                word2ph.append(len(phone)) #单个字的音素数量

            phones_list += phone # => ["zh","e","zh","en","sh","i","yi","g","e","h","ao","r","en"]
            tones_list += [int(tone)] * len(phone) # => [4, 4, 2, 2, 4, 4, 1, 3, 3, 3, 3, 3, 3] 相当于是对应着上面音素的音调
    #[音素], [声调], [每个字的音素数量]
    #["zh","e","zh","en","sh","i","yi","g","e","h","ao","r","en"], [4, 4, 2, 2, 4, 4, 1, 3, 3, 3, 3, 3, 3], [2, 2, 2, 1, 2, 2, 2]
    return phones_list, tones_list, word2ph


def text_normalize(text):
    numbers = re.findall(r"\d+(?:\.?\d+)?", text)
    for number in numbers:
        text = text.replace(number, cn2an.an2cn(number), 1)
    text = replace_punctuation(text)
    return text


def get_bert_feature(text, word2ph):
    from text import chinese_bert

    return chinese_bert.get_bert_feature(text, word2ph)


if __name__ == "__main__":
    from text.chinese_bert import get_bert_feature

    text = "啊！但是《原神》是由,米哈\游自主，  [研发]的一款全.新开放世界.冒险游戏"
    text = text_normalize(text)
    print(text)
    phones, tones, word2ph = g2p(text)
    bert = get_bert_feature(text, word2ph)

    print(phones, tones, word2ph, bert.shape)


# # 示例用法
# text = "这是一个示例文本：,你好！这是一个测试...."
# print(g2p_paddle(text))  # 输出: 这是一个示例文本你好这是一个测试
