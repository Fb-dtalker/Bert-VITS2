import torch
import sys
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("./bert/chinese-roberta-wwm-ext-large")

models = dict()


def get_bert_feature(text, word2ph, device=None):
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    if not device:
        device = "cuda"
    if device not in models.keys():
        models[device] = AutoModelForMaskedLM.from_pretrained(
            "./bert/chinese-roberta-wwm-ext-large"
        ).to(device)
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt") #将一句话的每个字转为编号的形式
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = models[device](**inputs, output_hidden_states=True)
        # res["hidden_stats"] => (embedding层output数量1+隐藏层各层输出数量, batch_size, seq_len, hidden_size)但是断点发现没有batch_size维
        # https://www.cnblogs.com/xiximayou/p/15016604.html
        # res["hidden_states"]是一个元组，里面[-3:-2]是bert最后的隐藏层输出结果，res["hidden_states"][-3:-2]=>[shape为(1, seq_len, hidden_size)的矩阵]
        # torch.cat(res["hidden_states"][-3:-2], -1) => shape为(1, seq_len, hidden_size)的矩阵
        # torch.cat(res["hidden_states"][-3:-2], -1)[0] => shape为(seq_len, hidden_size)的矩阵
        # 取得最后一层隐藏层的输出结果(batch_size, seq_len, hidden_size).cpu()
        # print(len(res["hidden_states"]))
        # print(res["hidden_states"][-3:-2][0].shape)
        # print(torch.cat(res["hidden_states"][-3:-2], -1).shape)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()

    assert len(word2ph) == len(text) + 2 #首先word2ph数量是长度
    word2phone = word2ph
    phone_level_feature = []
    for i in range(len(word2phone)):
        # res里面的是对每个字的结果，对应多音节的字，要以相同数据扩展到多个音节
        repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    # cat后的phone_level_feature.shape: (音节序列长度, hidden_size=1024)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    # 返回bert处理后对应每个字在扩展长度为音素后的结果
    return phone_level_feature.T


if __name__ == "__main__":
    import torch

    word_level_feature = torch.rand(38, 1024)  # 12个词,每个词1024维特征
    word2phone = [
        1,
        2,
        1,
        2,
        2,
        1,
        2,
        2,
        1,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        2,
        1,
        1,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        1,
    ]

    # 计算总帧数, 即音素数量
    total_frames = sum(word2phone)
    print(total_frames)
    print(word_level_feature.shape)
    print(word2phone)
    phone_level_feature = []
    for i in range(len(word2phone)):
        print(word_level_feature[i].shape)

        # 对每个词重复word2phone[i]次
        repeat_feature = word_level_feature[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    print(phone_level_feature.shape)  # torch.Size([36, 1024])

    # get_bert_feature("谢谢你", [0, 2, 2, 1, 0])