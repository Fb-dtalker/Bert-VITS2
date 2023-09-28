from text import chinese, japanese, cleaned_text_to_sequence


language_module_map = {"ZH": chinese, "JP": japanese}


def clean_text(text, language):
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text) #数字替换为汉字表达、符号替换为对应英文或者'符号 => 全汉字+英文符号的内容
    #[音素], [声调], [每个字的音素数量]
    #["zh","e","zh","en","sh","i","yi","g","e","h","ao","r","en"], [4, 4, 2, 2, 4, 4, 1, 3, 3, 3, 3, 3, 3], [2, 2, 2, 1, 2, 2, 2]
    phones, tones, word2ph = language_module.g2p(norm_text)
    return norm_text, phones, tones, word2ph


def clean_text_bert(text, language):
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(norm_text)
    bert = language_module.get_bert_feature(norm_text, word2ph)
    return phones, tones, bert


def text_to_sequence(text, language):
    norm_text, phones, tones, word2ph = clean_text(text, language)
    return cleaned_text_to_sequence(phones, tones, language)


if __name__ == "__main__":
    pass
