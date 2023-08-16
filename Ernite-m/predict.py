from model import NetWork
import os
import paddle
from data import test_dataloader

# 声明模型
model = NetWork("image")
print(model)

# 根据实际运行情况，更换加载的参数路径
params_path = "checkpoint/model_best.pdparams"
if params_path and os.path.isfile(params_path):
    # 加载模型参数
    state_dict = paddle.load(params_path)
    model.set_dict(state_dict)
    print("Loaded parameters from %s" % params_path)

results = []
# 切换model模型为评估模式，关闭dropout等随机因素
model.eval()
count = 0
for batch in test_dataloader:
    count += 1
    cap_batch, img_batch, qCap_batch, qImg_batch = batch
    logits = model(qCap=qCap_batch, qImg=qImg_batch, caps=cap_batch, imgs=img_batch)
    # 预测分类
    probs = F.softmax(logits, axis=-1)
    label = paddle.argmax(probs, axis=1).numpy()
    results += label.tolist()
    print(count)
print(results[:5])
print(len(results))

import pandas as pd

# id/label
# 字典中的key值即为csv中的列名
id_list = range(len(results))
print(id_list)
frame = pd.DataFrame({"id": id_list, "label": results})
frame.to_csv("result.csv", index=False, sep=",")
