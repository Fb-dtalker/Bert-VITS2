from text.symbols import *


_symbol_to_id = {s: i for i, s in enumerate(symbols)}


def cleaned_text_to_sequence(cleaned_text, tones, language):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
       将音素转换为音素对应的id，声调根据语言类型进行偏移，生成一个对应音素的语言id的列表
       思考：时不时可以通过修改id结果来实现多语种能在一句话里面进行TTS
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    phones = [_symbol_to_id[symbol] for symbol in cleaned_text] #将音素转为数字编号
    tone_start = language_tone_start_map[language] #声调开始下标
    tones = [i + tone_start for i in tones] #将声调抽象为偏移开始+当前语言声调排序，目的应该是为了兼容多语言的声调
    lang_id = language_id_map[language] #语言换为语言id
    lang_ids = [lang_id for i in phones] #建立一个和音素一样长度但是里面内容都是语言id的列表
    # [音素的数字编号], [经过语种偏移的声调下标], [长度和音素数字编号列表一样的但是里面是语种的id]
    # 音素转数字编号:[12, 1, 23, 32]
    # 声调下标偏移:[2 , 2,  3,  4]
    # 指定每个音素语种:[0 , 0,  0,  0]
    return phones, tones, lang_ids


def get_bert(norm_text, word2ph, language, device=None):
    '''
    norm_text:
    '''
    from .chinese_bert import get_bert_feature as zh_bert
    from .english_bert_mock import get_bert_feature as en_bert
    from .japanese_bert import get_bert_feature as jp_bert

    lang_bert_func_map = {"ZH": zh_bert, "EN": en_bert, "JP": jp_bert}
    bert = lang_bert_func_map[language](norm_text, word2ph, device) #内容文字, 音素数量, 使用gpu?
    return bert
