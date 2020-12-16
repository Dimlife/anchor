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
from anchor_ import anchor_text
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

# nlp = spacy.load('/home/jinzhiyu/anchor/en_core_web_lg-3.0.0a0/en_core_web_lg/en_core_web_lg-3.0.0a0')
# nlp = spacy.load('/home/jinzhiyu/anchor/en_core_web_sm-3.0.0a0/en_core_web_sm/en_core_web_sm-3.0.0a0')
nlp = spacy.load('/home/jinzhiyu/anchor/zh_core_web_sm-3.0.0a0/zh_core_web_sm/zh_core_web_sm-3.0.0a0')

data, labels = ['你好 上海', '你好 杨浦', '你好 北京', '你不好好 哥哥', '你好 菜鸟', '你好 不夜城', '你不好 不夜城', '你不好 哥', '你不好 背景'], [1, 1, 1, 0, 1, 1, 0, 0, 0]
# data, labels = load_polarity()
print(data, labels)
train, test, train_labels, test_labels = sklearn.model_selection.train_test_split(data, labels, test_size=.2, random_state=42)
train, val, train_labels, val_labels = sklearn.model_selection.train_test_split(train, train_labels, test_size=.1, random_state=42)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)
val_labels = np.array(val_labels)

vectorizer = CountVectorizer(min_df=1)
vectorizer.fit(train)
train_vectors = vectorizer.transform(train)
test_vectors = vectorizer.transform(test)
val_vectors = vectorizer.transform(val)

c = sklearn.linear_model.LogisticRegression()
# c = sklearn.ensemble.RandomForestClassifier(n_estimators=500, n_jobs=10)
c.fit(train_vectors, train_labels)
preds = c.predict(val_vectors)
print('Val accuracy', sklearn.metrics.accuracy_score(val_labels, preds))
def predict_lr(texts):
    return c.predict(vectorizer.transform(texts))

print(predict_lr('你好 朋友'))

# explainer = anchor_text.AnchorText(nlp, ['negative', 'positive'], use_unk_distribution=True)
#
# np.random.seed(1)
# text = '你好 好哥哥'
# pred = explainer.class_names[predict_lr([text])[0]]
# alternative = explainer.class_names[1 - predict_lr([text])[0]]
# print('Prediction: %s' % pred)
# exp = explainer.explain_instance(text, predict_lr, threshold=0.95)
#
# print('Anchor: %s' % (' AND '.join(exp.names())))
# print('Precision: %.2f' % exp.precision())
# print()
# print('Examples where anchor applies and model predicts %s:' % pred)
# print()
# print('\n'.join([x[0] for x in exp.examples(only_same_prediction=True)]))
# print()
# print('Examples where anchor applies and model predicts %s:' % alternative)
# print()
# print('\n'.join([x[0] for x in exp.examples(partial_index=0, only_different_prediction=True)]))

# # changing the distribution
# explainer = anchor_text.AnchorText(nlp, ['negative', 'positive'], use_unk_distribution=False, use_bert=False)
#
# np.random.seed(1)
# text = 'This is a good book .'
# pred = explainer.class_names[predict_lr([text])[0]]
# alternative =  explainer.class_names[1 - predict_lr([text])[0]]
# print('Prediction: %s' % pred)
# exp = explainer.explain_instance(text, predict_lr, threshold=0.95, verbose=False, use_proba=True)
#
#
# print('Anchor: %s' % (' AND '.join(exp.names())))
# print('Precision: %.2f' % exp.precision())
# print()
# print('Examples where anchor applies and model predicts %s:' % pred)
# print()
# print('\n'.join([x[0] for x in exp.examples(only_same_prediction=True)]))
# print()
# print('Examples where anchor applies and model predicts %s:' % alternative)
# print()
# print('\n'.join([x[0] for x in exp.examples(only_different_prediction=True)]))
#
# print('Partial anchor: %s' % (' AND '.join(exp.names(0))))
# print('Precision: %.2f' % exp.precision(0))
# print()
# print('Examples where anchor applies and model predicts %s:' % pred)
# print()
# print('\n'.join([x[0] for x in exp.examples(partial_index=0, only_same_prediction=True)]))
# print()
# print('Examples where anchor applies and model predicts %s:' % alternative)
# print()
# print('\n'.join([x[0] for x in exp.examples(partial_index=0, only_different_prediction=True)]))
#
#
explainer = anchor_text.AnchorText(nlp, ['negative', 'positive'], use_unk_distribution=False, use_bert=True)


np.random.seed(1)
text = '生命的真谛是 爱'
pred = explainer.class_names[predict_lr([text])[0]]
alternative = explainer.class_names[1 - predict_lr([text])[0]]
print('Prediction: %s' % pred)
b = time.time()
exp = explainer.explain_instance(text, predict_lr, threshold=0.95, verbose=False)
print('Time: %s' % (time.time() - b))

print('Anchor: %s' % (' AND '.join(exp.names())))
print('Precision: %.2f' % exp.precision())
print()
print('Examples where anchor applies and model predicts %s:' % pred)
print()
print('\n'.join([x[0] for x in exp.examples(only_same_prediction=True)]))
print()
print('Examples where anchor applies and model predicts %s:' % alternative)
print()
print('\n'.join([x[0] for x in exp.examples(only_different_prediction=True)]))

#
# print('Partial anchor: %s' % (' AND '.join(exp.names(0))))
# print('Precision: %.2f' % exp.precision(0))
# print()
# print('Examples where anchor applies and model predicts %s:' % pred)
# print()
# print('\n'.join([x[0] for x in exp.examples(partial_index=0, only_same_prediction=True)]))
# print()
# print('Examples where anchor applies and model predicts %s:' % alternative)
# print()
# print('\n'.join([x[0] for x in exp.examples(partial_index=0, only_different_prediction=True)]))
#
#
# np.random.seed(1)
# text = 'This is a good book .'
# pred = explainer.class_names[predict_lr([text])[0]]
# alternative =  explainer.class_names[1 - predict_lr([text])[0]]
# print('Prediction: %s' % pred)
# b = time.time()
# exp = explainer.explain_instance(text, predict_lr, threshold=0.95, verbose=False, onepass=True)
# print('Time: %s' % (time.time() - b))
#
# print('Anchor: %s' % (' AND '.join(exp.names())))
# print('Precision: %.2f' % exp.precision())
# print()
# print('Examples where anchor applies and model predicts %s:' % pred)
# print()
# print('\n'.join([x[0] for x in exp.examples(only_same_prediction=True)]))
# print()
# print('Examples where anchor applies and model predicts %s:' % alternative)
# print()
# print('\n'.join([x[0] for x in exp.examples(only_different_prediction=True)]))
#
#
# print('Partial anchor: %s' % (' AND '.join(exp.names(0))))
# print('Precision: %.2f' % exp.precision(0))
# print()
# print('Examples where anchor applies and model predicts %s:' % pred)
# print()
# print('\n'.join([x[0] for x in exp.examples(partial_index=2, only_same_prediction=True)]))
# print()
# print('Examples where anchor applies and model predicts %s:' % alternative)
# print()
# print('\n'.join([x[0] for x in exp.examples(partial_index=2, only_different_prediction=True)]))
#
