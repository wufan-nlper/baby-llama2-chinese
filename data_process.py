import json
import glob
import numpy as np
import pandas as pd

from tqdm import tqdm
from chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer


tokenizer = ChatGLMTokenizer(vocab_file="./chatglm_tokenizer/tokenizer.model")


def process_wiki_clean(json_p, bin_p):
    with open(json_p, "r") as f:
        data = json.load(f)
    doc_ids = []
    for line in tqdm(data):
        text = line["completion"]
        text_id = tokenizer.encode(text, add_special_tokens=False)
        text_id.append(tokenizer.special_tokens["<eos>"])
        if len(text_id) > 5:
            doc_ids += text_id
    arr = np.array(doc_ids, dtype=np.uint16)
    with open(bin_p, "wb") as f:
        f.write(arr.tobytes())


def process_medical(json_p, bin_p):
    f = open(json_p, "r")
    doc_ids = []
    while True:
        line = f.readline()
        if not line:
            break
        line = json.loads(line)
        text = line["text"]
        text_id = tokenizer.encode(text, add_special_tokens=False)
        text_id.append(tokenizer.special_tokens["<eos>"])
        if len(text_id) > 5:
            doc_ids += text_id
    arr = np.array(doc_ids, dtype=np.uint16)
    with open(bin_p, "wb") as f:
        f.write(arr.tobytes())


def sft_to_pretrain():
    df = pd.read_csv("/disk/wufan/gpt_data/medical_qa_144w.csv")
    doc_ids = []
    for _, q, a in tqdm(df.itertuples()):
        q_id = tokenizer.encode(q, add_special_tokens=False)
        a_id = tokenizer.encode(a, add_special_tokens=False)
        #
        print(q)
        print(a)
        print("-----")
        text_id = q_id + a_id + [tokenizer.special_tokens["<eos>"]]
        if len(text_id) > 5:
            doc_ids += text_id
    arr = np.array(doc_ids, dtype=np.uint16)
    print(arr.shape)
    with open("/disk/wufan/gpt_data/medical_qa.bin", "wb") as f:
        f.write(arr.tobytes())


def sft_process():
    with open("/disk/wufan/gpt_data/alpaca_gpt4_data_zh.json", "r") as f:
        data = json.load(f)
    #
    q_lst = []
    a_lst = []
    for per in data:
        q = per["instruction"]
        i = per["input"]
        a = per["output"]
        q = q + i
        if len(q) < 10 or len(a) < 5:
            continue
        if len(q) > 256 or len(a) > 256:
            continue
        q_lst.append(q)
        a_lst.append(a)
    #
    # with open('../track1/train_valid.json','r') as f:
    #     data = json.load(f)
    # #
    # for l in data:
    #     q_lst.append(l['question'])
    #     a_lst.append(l['answer'])
    #
    f = open("/disk/wufan/gpt_data/Belle_open_source_1M.json", "r")

    # s
    while True:
        line = f.readline()
        if not line:
            break
        per = json.loads(line)
        q = per["instruction"]
        i = per["input"]
        a = per["output"]
        q = q + i
        if len(q) < 10 or len(a) < 5:
            continue
        if len(q) > 256 or len(a) > 256:
            continue
        q_lst.append(q)
        a_lst.append(a)
    df = pd.DataFrame(columns=["prompt", "answer"])
    df["prompt"] = q_lst
    df["answer"] = a_lst
    df.to_csv("data/sft_data.csv", index=False)
    print(df)


def process_baidu(json_p, bin_p):
    f = open(json_p, "r")
    cnt = 0
    token = 0
    doc_ids = []
    while True:
        line = f.readline()
        if not line:
            break
        line = json.loads(line)
        text = ""
        try:
            text += line["title"] + "：" + line["summary"]
        except:
            pass
        for per in line["sections"]:
            text += per["title"] + "：" + per["content"] + "。"
        text_id = tokenizer.encode(text, add_special_tokens=False)
        text_id.append(tokenizer.special_tokens["<eos>"])
        if len(text_id) > 5:
            doc_ids += text_id
        cnt += 1
        if cnt % 10000 == 0:
            print(cnt)

    arr = np.array(doc_ids, dtype=np.uint16)
    print(arr.shape)
    with open(bin_p, "wb") as f:
        f.write(arr.tobytes())


if __name__ == "__main__":
    pass
    process_wiki_clean()
    process_medical("/disk/wufan/gpt_data/medical_book_zh.json", "book")
    process_medical("/disk/wufan/gpt_data/train_encyclopedia.json", "encyclopedia")
    sft_to_pretrain()
    sft_process()
    # process_baidu()
    data_path_list = [
        "/disk/wufan/gpt_data/baidubaike_563w.bin",
        "/disk/wufan/gpt_data/medical_book.bin",
        # "/disk/wufan/gpt_data/medical_encyclopedia.bin",
        "/disk/wufan/gpt_data/wiki.bin",
    ]
    data_lst = []
    for data_path in data_path_list:
        with open(data_path, "rb") as f:
            data = np.fromfile(f, dtype=np.uint16)
            data_lst.append(data)
    arr = np.concatenate(data_lst)
    print(arr.shape)
    with open("/disk/wufan/gpt_data/pretrain_data.bin", "wb") as f:
        f.write(arr.tobytes())
