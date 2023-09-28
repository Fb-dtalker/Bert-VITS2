import json
from collections import defaultdict
from random import shuffle
from typing import Optional

from tqdm import tqdm
import click
from text.cleaner import clean_text


@click.command()
@click.option(
    "--transcription-path",
    default="filelists/genshin.list",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--cleaned-path", default=None)
@click.option("--train-path", default="filelists/train.list")
@click.option("--val-path", default="filelists/val.list")
@click.option(
    "--config-path",
    default="configs/config.json",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--val-per-spk", default=4)
@click.option("--max-val-total", default=8)
@click.option("--clean/--no-clean", default=True)
def main(
    transcription_path: str,
    cleaned_path: Optional[str],
    train_path: str,
    val_path: str,
    config_path: str,
    val_per_spk: int,
    max_val_total: int,
    clean: bool,
):
    if cleaned_path is None:
        cleaned_path = transcription_path + ".cleaned"

    if clean:
        out_file = open(cleaned_path, "w", encoding="utf-8")
        for line in tqdm(open(transcription_path, encoding="utf-8").readlines()):
            try:
                # "音频位置|说话者|语言|话语内容"
                utt, spk, language, text = line.strip().split("|")
                # 转换成全汉字和英语标点的句子, [音素], [声调], [每个字的音素数量]
                # "这真是一个好人", ["zh","e","zh","en","sh","i","yi","g","e","h","ao","r","en"], [4, 4, 2, 2, 4, 4, 1, 3, 3, 3, 3, 3, 3], [2, 2, 2, 1, 2, 2, 2]
                norm_text, phones, tones, word2ph = clean_text(text, language)
                # "音频位置|说话者名字|语言|转换成全汉字和英语标点的句子|空格连接的音素|空格连接的声调|空格连接的音素数量"
                out_file.write(
                    "{}|{}|{}|{}|{}|{}|{}\n".format(
                        utt,
                        spk,
                        language,
                        norm_text,
                        " ".join(phones),
                        " ".join([str(i) for i in tones]),
                        " ".join([str(i) for i in word2ph]),
                    )
                )
            except Exception as error:
                print("err!", line, error)

        out_file.close()

        transcription_path = cleaned_path # 用清理后内容的文本路径覆盖要进行处理的文本路径

    spk_utt_map = defaultdict(list)
    spk_id_map = {}
    current_sid = 0

    # transcription_path指向的文件中每一行都是cleaned的内容："音频位置|说话者名字|语言|转换成全汉字和英语标点的句子|空格连接的音素|空格连接的声调|空格连接的音素数量"
    with open(transcription_path, encoding="utf-8") as f:
        for line in f.readlines():
            utt, spk, language, text, phones, tones, word2ph = line.strip().split("|")
            #{
            #   "spk1":["一行","另一行"...]
            # }
            spk_utt_map[spk].append(line)

            # 人名->数字id的映射
            if spk not in spk_id_map.keys():
                spk_id_map[spk] = current_sid
                current_sid += 1

    train_list = []
    val_list = []

    # 为每个角色划分训练集和验证集，然后把全部的训练集和验证集分别拼接层列表，
    # 列表里面的元素是："音频位置|说话者名字|语言|转换成全汉字和英语标点的句子|空格连接的音素|空格连接的声调|空格连接的音素数量"
    for spk, utts in spk_utt_map.items():
        shuffle(utts)
        val_list += utts[:val_per_spk]
        train_list += utts[val_per_spk:]

    if len(val_list) > max_val_total:
        train_list += val_list[max_val_total:]
        val_list = val_list[:max_val_total]

    # 将分出来的训练集和验证集分行写到训练集文本和验证集文本中
    with open(train_path, "w", encoding="utf-8") as f:
        for line in train_list:
            f.write(line)

    with open(val_path, "w", encoding="utf-8") as f:
        for line in val_list:
            f.write(line)

    # 读取配置文件，然后更新说话者列表，再写回保存
    config = json.load(open(config_path, encoding="utf-8"))
    config["data"]["spk2id"] = spk_id_map
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
