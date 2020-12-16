import requests
import json
from tqdm import tqdm
import pandas as pd
from lime import lime_text
from lime.lime_text import LimeTextExplainer
from anchor import anchor_text
import numpy as np


def send_request(data, dmid_choose=0):
    """

    :param data: string_list
    :param dmid_choose: int
    :return: np_array of real numbers
    """
    batch = 2000
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
        print(i)
        print(len(data))
        cur_data = data[i * batch: (i + 1) * batch]
        print(len(cur_data))
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
                my_total_0.append(1 if score > 0.7 else 0)
                # print(score)
        except json.decoder.JSONDecodeError:
            print(request_data)
        # print('my_total_0', my_total_0)
    return np.array(my_total_0)

input = ['你好 陌生人']
print(send_request(input))