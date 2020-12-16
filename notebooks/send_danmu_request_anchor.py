import requests
import json
from tqdm import tqdm
import pandas as pd
from lime import lime_text
from lime.lime_text import LimeTextExplainer
from anchor import anchor_text
import numpy as np


def send_request(data):
    """

    :param data: string_list
    :param dmid_choose: int
    :return: np_array of real numbers
    """
    batch = 10
    dmid_choose = 0
    request_data = {
        "dmids": [],
        "danmaku": [],
        "ctime": [],
        "parts": [],
        "modes": [],
        "progress": [],
        "dur": [],
        "report_rate": [],
        "sexes": [],
        "type": "live"
    }
    my_total_0 = []
    my_total_2 = []
    my_total_4 = []
    my_total_6 = []
    my_total_8 = []

    # print(data)
    for i in tqdm(range(len(data) // batch + 1)):
        cur_data = data[i * batch: (i + 1) * batch]
        if len(cur_data) == 0:
            continue
        sex = [1] * len(cur_data)
        report_rate = [1] * len(cur_data)
        ctime = [1595540157] * len(cur_data)
        cur_msg = [item for item in cur_data]
        cur_part = [11] * len(cur_data)
        cur_mode = [1] * len(cur_data)
        cur_progress = [1] * len(cur_data)
        cur_dur = [1] * len(cur_data)
        request_data['dmids'] = [dmid_choose] * len(cur_data)
        request_data['danmaku'] = cur_msg
        request_data['ctime'] = ctime
        request_data['parts'] = cur_part
        request_data['modes'] = cur_mode
        request_data['progress'] = cur_progress
        request_data['dur'] = cur_dur
        request_data['report_rate'] = report_rate
        request_data['sexes'] = sex
        try:
            f_my8 = requests.post('http://deeplearn.bilibili.co/dl/api/dmscore/v1', json=request_data).json()
            for score in f_my8['scores']:
                # my_total_0.append([1 - score, score])
                my_total_0.append(1 if score > 0.67 else 0)
        except json.decoder.JSONDecodeError:
            print(request_data)

    return np.array(my_total_0)


def word_split(input_str):
    url = 'http://deeplearn.bilibili.co/nlp/cws'
    request_data = {"inputs": [input_str]}
    response = requests.post(url, json=request_data)
    return response.json()


def list_round(weights, digit=3):
    return [(weight[0], round(weight[1], digit)) for weight in weights]


def word_explain(input_str, d_choose):
    explainer = LimeTextExplainer(class_names=['negative', 'positive'])
    input = word_split(input_str)['results'][0]
    send_func = lambda x: send_request(x, dmid_choose=d_choose)
    exp = explainer.explain_instance(input, send_func, num_features=10)
    return list_round(exp.as_list())


if __name__ == '__main__':
    while True:
        input_string = input('输入')
        if input_string == ' ':
            break
        print(send_request([input_string], 0))
        # print(word_explain(input_string, 4))
        # print(word_explain(input_string, 6))
