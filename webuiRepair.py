# flake8: noqa: E402

import sys, os
import logging

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO, format="| %(name)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

import torch
import argparse
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import cleaned_text_to_sequence, get_bert
from text.cleaner import clean_text
import gradio as gr
import webbrowser

net_g = None

if sys.platform == "darwin" and torch.backends.mps.is_available():
    device = "mps"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
else:
    device = "cuda"

def get_text(text, language_str, hps):
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    bert = get_bert(norm_text, word2ph, language_str, device)
    del word2ph
    assert bert.shape[-1] == len(phone), phone

    if language_str == "ZH":
        bert = bert
        ja_bert = torch.zeros(768, len(phone))
    elif language_str == "JA":
        ja_bert = bert
        bert = torch.zeros(1024, len(phone))
    else:
        bert = torch.zeros(1024, len(phone))
        ja_bert = torch.zeros(768, len(phone))

    assert bert.shape[-1] == len(
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"

    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)
    return bert, ja_bert, phone, tone, language

def get_text_pinyin(text, language_str, hps, phoneOverride = None, toneOverride = None):
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    if phoneOverride is not None:
        phone = phoneOverride
        tone = toneOverride
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    bert = get_bert(norm_text, word2ph, language_str, device)
    del word2ph
    assert bert.shape[-1] == len(phone), phone

    if language_str == "ZH":
        bert = bert
        ja_bert = torch.zeros(768, len(phone))
    elif language_str == "JA":
        ja_bert = bert
        bert = torch.zeros(1024, len(phone))
    else:
        bert = torch.zeros(1024, len(phone))
        ja_bert = torch.zeros(768, len(phone))

    assert bert.shape[-1] == len(
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"

    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)
    return bert, ja_bert, phone, tone, language

def infer(text, sdp_ratio, noise_scale, noise_scale_w, length_scale, sid, language):
    global net_g
    bert, ja_bert, phones, tones, lang_ids = get_text(text, language, hps)
    with torch.no_grad():
        x_tst = phones.to(device).unsqueeze(0)
        tones = tones.to(device).unsqueeze(0)
        lang_ids = lang_ids.to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        ja_bert = ja_bert.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        del phones
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(device)
        audio = (
            net_g.infer(
                x_tst,
                x_tst_lengths,
                speakers,
                tones,
                lang_ids,
                bert,
                ja_bert,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )[0][0, 0]
            .data.cpu()
            .float()
            .numpy()
        )
        del x_tst, tones, lang_ids, bert, x_tst_lengths, speakers
        return audio

def infer_pinyin_second(text, sdp_ratio, noise_scale, noise_scale_w, length_scale, sid, language, merge_pinyin):
    phonesOverride, tonesOverride = split_phones_tones(merge_pinyin)
    audio, merge_pinyin = infer_pinyin_first(text, sdp_ratio, noise_scale, noise_scale_w, length_scale, sid, language, phonesOverride, tonesOverride)
    return audio, merge_pinyin

def infer_pinyin_first(text, sdp_ratio, noise_scale, noise_scale_w, length_scale, sid, language, phonesOverride = None, tonesOverride = None):
    global net_g
    bert, ja_bert, phones, tones, lang_ids = get_text_pinyin(text, language, hps, phonesOverride, tonesOverride)

    # 清理出文字版的音素和声调
    _, phone_str_list, tone_int_list, _ = clean_text(text, language)
    if phonesOverride is not None:
        phone_str_list = phonesOverride
        tone_int_list = tonesOverride
    merge_pinyin = merge_phones_tones(phone_str_list, tone_int_list)

    with torch.no_grad():
        x_tst = phones.to(device).unsqueeze(0)
        tones = tones.to(device).unsqueeze(0)
        lang_ids = lang_ids.to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        ja_bert = ja_bert.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        del phones
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(device)
        audio = (
            net_g.infer(
                x_tst,
                x_tst_lengths,
                speakers,
                tones,
                lang_ids,
                bert,
                ja_bert,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )[0][0, 0]
            .data.cpu()
            .float()
            .numpy()
        )
        del x_tst, tones, lang_ids, bert, x_tst_lengths, speakers
        return audio, merge_pinyin

def merge_phones_tones(phones, tones):
    index = 0
    merge = []
    while(index < len(phones)):
        merge.append(phones[index] + str(tones[index]))
        index += 1
    return " ".join(merge)

def split_phones_tones(merge_text):
    split = merge_text.split(" ")
    phones = []
    tones = []
    for phone_tone in split:
            phones.append(phone_tone[0:-1]) #取出拼音部分
            tones.append(int(phone_tone[-1:])) #取出声调部分
    return phones, tones

def tts_fn(
    text, speaker, sdp_ratio, noise_scale, noise_scale_w, length_scale, language
):
    with torch.no_grad():
        audio = infer(
            text,
            sdp_ratio=sdp_ratio,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            length_scale=length_scale,
            sid=speaker,
            language=language,
        )
        torch.cuda.empty_cache()
    return "Success", (hps.data.sampling_rate, audio)

def tts_fn_pinyin_first(
    text, speaker, sdp_ratio, noise_scale, noise_scale_w, length_scale, language
):
    with torch.no_grad():
        audio, pinyin = infer_pinyin_first(
            text,
            sdp_ratio=sdp_ratio,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            length_scale=length_scale,
            sid=speaker,
            language=language,
        )
        torch.cuda.empty_cache()
    return "Success with first!", (hps.data.sampling_rate, audio), pinyin

def tts_fn_pinyin_second(
    text, speaker, sdp_ratio, noise_scale, noise_scale_w, length_scale, language, merge_pinyin
):
    with torch.no_grad():
        audio, merge_pinyin = infer_pinyin_second(
            text,
            sdp_ratio=sdp_ratio,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            length_scale=length_scale,
            sid=speaker,
            language=language,
            merge_pinyin=merge_pinyin
        )
        torch.cuda.empty_cache()
    return "Success with second!", (hps.data.sampling_rate, audio), merge_pinyin

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", default="./logs/as/G_8000.pth", help="path of your model"
    )
    parser.add_argument(
        "-c",
        "--config",
        default="./configs/config.json",
        help="path of your config file",
    )
    parser.add_argument(
        "--share", default=False, help="make link public", action="store_true"
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="enable DEBUG-LEVEL log"
    )

    args = parser.parse_args()
    if args.debug:
        logger.info("Enable DEBUG-LEVEL log")
        logging.basicConfig(level=logging.DEBUG)
    hps = utils.get_hparams_from_file(args.config)

    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else (
            "mps"
            if sys.platform == "darwin" and torch.backends.mps.is_available()
            else "cpu"
        )
    )
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).to(device)
    _ = net_g.eval()

    _ = utils.load_checkpoint(args.model, net_g, None, skip_optimizer=True)

    speaker_ids = hps.data.spk2id
    speakers = list(speaker_ids.keys())
    languages = ["ZH", "JP"]
    with gr.Blocks() as app:
        with gr.Row():
            with gr.Column():
                text = gr.TextArea(
                    label="Text",
                    placeholder="Input Text Here",
                    value="吃葡萄不吐葡萄皮，不吃葡萄倒吐葡萄皮。",
                )
                speaker = gr.Dropdown(
                    choices=speakers, value=speakers[0], label="Speaker"
                )
                sdp_ratio = gr.Slider(
                    minimum=0, maximum=1, value=0.2, step=0.1, label="SDP Ratio"
                )
                noise_scale = gr.Slider(
                    minimum=0.1, maximum=2, value=0.6, step=0.1, label="Noise Scale"
                )
                noise_scale_w = gr.Slider(
                    minimum=0.1, maximum=2, value=0.8, step=0.1, label="Noise Scale W"
                )
                length_scale = gr.Slider(
                    minimum=0.1, maximum=2, value=1, step=0.1, label="Length Scale"
                )
                language = gr.Dropdown(
                    choices=languages, value=languages[0], label="Language"
                )
                btn = gr.Button("Generate!", variant="primary")

                pinyin_text = gr.TextArea(
                    label="Text",
                    placeholder="Input pinyin here",
                    value="zh3 e3 li3 f4ang4 pin1 yin1",
                )
                btn2 = gr.Button("Generate with repaired pinyin no override!", variant="primary")
                btn3 = gr.Button("Generate with repaired pinyin do override!", variant="primary")
            with gr.Column():
                text_output = gr.Textbox(label="Message")
                audio_output = gr.Audio(label="Output Audio")

        btn.click(
            tts_fn,
            inputs=[
                text,
                speaker,
                sdp_ratio,
                noise_scale,
                noise_scale_w,
                length_scale,
                language,
            ],
            outputs=[text_output, audio_output],
        )

        btn2.click(
            tts_fn_pinyin_first,
            inputs=[
                text,
                speaker,
                sdp_ratio,
                noise_scale,
                noise_scale_w,
                length_scale,
                language,
            ],
            outputs=[text_output, audio_output, pinyin_text],
        )

        btn3.click(
            tts_fn_pinyin_second,
            inputs = [
                text,
                speaker,
                sdp_ratio,
                noise_scale,
                noise_scale_w,
                length_scale,
                language,
                pinyin_text,
            ],
            outputs=[text_output, audio_output, pinyin_text],
        )

    webbrowser.open("http://127.0.0.1:7861")
    app.launch(share=args.share)
