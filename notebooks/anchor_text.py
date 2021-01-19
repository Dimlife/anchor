# -*- coding: utf-8 -*-
import os
import os.path
import numpy as np
import sklearn
import sklearn.model_selection
import sklearn.linear_model
import sklearn.ensemble
import spacy
import sys
from sklearn.feature_extraction.text import CountVectorizer
from my_anchor import anchor_text
import requests
import json
from tqdm import tqdm
import time

def load_polarity(path='./rt-polaritydata'):
    data = []
    labels = []
    f_names = ['rt-polarity.neg', 'rt-polarity.pos']
    for (l, f) in enumerate(f_names):
        for line in open(os.path.join(path, f), 'rb'):
            try:
                line.decode('utf8')
            except:
                continue
            data.append(line.strip())
            labels.append(l)
    return data, labels

def predict_lr(data, dmid_choose=0):
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
        cur_data = data[i * batch: (i + 1) * batch]
        if len(cur_data) == 1:
            my_total_0.append(predict_lr([cur_data[0]] * 2, dmid_choose)[0])
            continue
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

nlp = spacy.load('/home/jinzhiyu/anchor/zh_core_web_sm-3.0.0a0/zh_core_web_sm/zh_core_web_sm-3.0.0a0')

'''
# data, labels = ['你好 上海', '你好 杨浦', '你好 北京', '你不好好 哥哥', '你好 菜鸟', '你好 不夜城', '你不好 不夜城', '你不好 哥', '你不好 背景'], [1, 1, 1, 0, 1, 1, 0, 0, 0]
# # data, labels = load_polarity()
# print(data, labels)
# train, test, train_labels, test_labels = sklearn.model_selection.train_test_split(data, labels, test_size=.2, random_state=42)
# train, val, train_labels, val_labels = sklearn.model_selection.train_test_split(train, train_labels, test_size=.1, random_state=42)
# train_labels = np.array(train_labels)
# test_labels = np.array(test_labels)
# val_labels = np.array(val_labels)
# 
# vectorizer = CountVectorizer(min_df=1)
# vectorizer.fit(train)
# train_vectors = vectorizer.transform(train)
# test_vectors = vectorizer.transform(test)
# val_vectors = vectorizer.transform(val)
# 
# c = sklearn.linear_model.LogisticRegression()
# # c = sklearn.ensemble.RandomForestClassifier(n_estimators=500, n_jobs=10)
# c.fit(train_vectors, train_labels)
# preds = c.predict(val_vectors)
# print('Val accuracy', sklearn.metrics.accuracy_score(val_labels, preds))
# def predict_lr(texts):
#     # 输入为 string 的 list
#     # 输出为 0/1    的 list
#     texts = [text.replace(' ', '') for text in texts]
#     print(texts)
#     return c.predict(vectorizer.transform(texts))
# 
# print(predict_lr(['你好 朋友', '你好 兄弟']))
# print(type(predict_lr(['你好 朋友', '你好 兄弟'])))
'''


explainer = anchor_text.AnchorText(nlp, ['negative', 'positive'], use_unk_distribution=False, use_bert=True)
np.random.seed(1)
text = '老 子 爱 死 你 了 大 笨 蛋'
print(explainer.class_names[0])
pred = explainer.class_names[int(predict_lr([text] * 10)[0])]
alternative = explainer.class_names[1 - predict_lr([text] * 10)[0]]
b = time.time()
exp = explainer.explain_instance(text, predict_lr, threshold=0.95, verbose=False)
print('Time: %s' % (time.time() - b))
print('Prediction: %s' % pred)
print('Anchor: %s' % (' AND '.join(exp.names())))
print('Precision: %.2f' % exp.precision())
print()
print('Examples where my_anchor applies and model predicts %s:' % pred)
print()
print('\n'.join([x[0] for x in exp.examples(only_same_prediction=True)]))
print()
print('Examples where my_anchor applies and model predicts %s:' % alternative)
print()
print('\n'.join([x[0] for x in exp.examples(only_different_prediction=True)]))
