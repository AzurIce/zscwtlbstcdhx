from paddle.vision import models
import paddle
from paddlenlp.transformers import ErnieMModel,ErnieMTokenizer
from paddle.nn import functional as F
from paddle import nn
import matplotlib.pyplot as plt
import numpy as np

class EncoderCNN(nn.Layer):
    def __init__(self, resnet_arch = 'resnet101'):
        super(EncoderCNN, self).__init__()
        if resnet_arch == 'resnet101':
            resnet = models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2D((1, 1))
    def forward(self, images, features='pool'):
        out = self.resnet(images)
        if features == 'pool':
            out = self.adaptive_pool(out)
            out = paddle.reshape(out, (out.shape[0],out.shape[1]))
        return out

class NetWork(nn.Layer):
    def __init__(self, mode):
        super(NetWork, self).__init__()
        self.mode = mode           
        self.ernie = ErnieMModel.from_pretrained('ernie-m-base')
        self.tokenizer = ErnieMTokenizer.from_pretrained('ernie-m-base')
        self.resnet = EncoderCNN()
        self.classifier1 = nn.Linear(2*(768+2048),1024)
        self.classifier2 = nn.Linear(1024,3)
        self.attention_text = nn.MultiHeadAttention(768,16)
        self.attention_image = nn.MultiHeadAttention(2048,16)
        if self.mode == 'text':
            self.classifier = nn.Linear(768,3)
        self.resnet.eval()

    def forward(self,qCap,qImg,caps,imgs):
        self.resnet.eval()
        encode_dict_qcap = self.tokenizer(text = qCap ,max_length = 128 ,truncation=True, padding='max_length')
        input_ids_qcap = encode_dict_qcap['input_ids']
        input_ids_qcap = paddle.to_tensor(input_ids_qcap)
        qcap_feature, pooled_output= self.ernie(input_ids_qcap) #(b,length,dim)
        if self.mode == 'text':
            logits = self.classifier(qcap_feature[:,0,:].squeeze(1))
            return logits
        caps_feature = []
        for i,caption in enumerate (caps):
            encode_dict_cap = self.tokenizer(text = caption ,max_length = 128 ,truncation=True, padding='max_length')
            input_ids_caps = encode_dict_cap['input_ids']
            input_ids_caps = paddle.to_tensor(input_ids_caps)
            cap_feature, pooled_output= self.ernie(input_ids_caps) #(b,length,dim)
            caps_feature.append(cap_feature)
        caps_feature = paddle.stack(caps_feature,axis=0) #(b,num,length,dim)
        caps_feature = caps_feature.mean(axis=1)#(b,length,dim)
        caps_feature = self.attention_text(qcap_feature,caps_feature,caps_feature) #(b,length,dim)
        imgs_features = []
        for img in imgs:
            imgs_feature = self.resnet(img) #(length,dim)
            imgs_features.append(imgs_feature)
        imgs_features = paddle.stack(imgs_features,axis=0) #(b,length,dim)
        qImg_features = []
        for qImage in qImg:
            qImg_feature = self.resnet(qImage.unsqueeze(axis=0)) #(1,dim)
            qImg_features.append(qImg_feature)
        qImg_feature = paddle.stack(qImg_features,axis=0) #(b,1,dim)
        imgs_features = self.attention_image(qImg_feature,imgs_features,imgs_features) #(b,1,dim)
        # [1, 128, 768] [1, 128, 768] [1, 1, 2048] [1, 1, 2048] origin
        # print(qcap_feature.shape,caps_feature.shape,qImg_feature.shape,imgs_features.shape)
        # print((qcap_feature[:,0,:].shape,caps_feature[:,0,:].shape,qImg_feature.squeeze(1).shape,imgs_features.squeeze(1).shape))
        # ([1,768], [1 , 768], [1, 2048], [1,  2048])
        feature = paddle.concat(x=[qcap_feature[:,0,:], caps_feature[:,0,:], qImg_feature.squeeze(1), imgs_features.squeeze(1)], axis=-1) 
        logits = self.classifier1(feature)
        logits = self.classifier2(logits)
        return logits